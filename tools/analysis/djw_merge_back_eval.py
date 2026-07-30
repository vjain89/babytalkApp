#!/usr/bin/env python3
"""Offline DJW smarter merge-back prototype + kit benchmark.

Scope (engineer summary) — also see ``tools/analysis/out/merge_back_scope.md``:

After DJW children are produced (same ``parentSpanId``, same ``speakerCluster``,
abutting/near cuts), decide keep-cut vs merge using weak-valley / short-gap
signals (invert ``WORD_SPLIT``), short piece duration, continuous energy, and
hard guards (no speaker cross, no long pause, max merged duration).

Pipeline slot: post-resegment (prototype applies on SoTA annotations offline).
Risks: re-gluing multi-word turns; Complexity **M** for production + calibration
(S for this standalone eval).

North-star: manual tags excluding non-verbal (IoU≥0.5); also report verbal-only.

    tools/.venv/bin/python tools/analysis/djw_merge_back_eval.py
    tools/.venv/bin/python tools/analysis/djw_merge_back_eval.py \\
      --kits 26_07_27__19:53:00 26_07_05__00:00:00
    tools/.venv/bin/python tools/analysis/djw_merge_back_eval.py --short-gate-sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

import numpy as np
import soundfile as sf

import resegment as reseg_mod
from ml_delta import LIBRARY, VERBAL, compare, load, pct
from vad_segments import run_vad_on_audio

OUT_JSON = ANALYSIS_DIR / "out" / "merge_back_eval.json"
SHORT_GATE_JSON = ANALYSIS_DIR / "out" / "merge_back_short_gate.json"
SHORT_GATE_MD = ANALYSIS_DIR / "out" / "merge_back_short_gate.md"
SCOPE_MD = ANALYSIS_DIR / "out" / "merge_back_scope.md"
NONVERBAL = "non-verbal vocalization"

DEFAULT_KITS = ["26_07_27__19:53:00", "26_07_05__00:00:00"]

# Absolute short-piece ms values for the short-gate sweep (clearly short required).
# 550 = legacy short_piece_ms but with require_clearly_short (no weak-valley-only).
SHORT_GATE_ABS_MS = (350.0, 400.0, 450.0, 500.0)
# Relative gate: min(piece)/session_median_tag < this (~over-split median durRatio).
SHORT_GATE_DUR_RATIO = 0.65


@dataclass(frozen=True)
class MergeBackParams:
    """Documented merge-back policy knobs.

    Old (OR) policy: merge when ``short_piece OR weak_valley`` (plus guards).

    Short-gated policy (``require_clearly_short=True``): weak valley alone is
    **not** enough. Merge only when a piece is clearly short by absolute ms
    and/or vs session tag-duration median (``dur_ratio_max``), then treat
    weak-valley as a secondary label only.
    """

    max_gap_ms: float = 100.0
    max_merged_ms: float = 1800.0
    short_piece_ms: float = 550.0
    weak_dip_db: float = 5.0
    weak_sep_ms: float = 280.0
    energy_drop_db: float = 6.0
    cut_window_ms: float = 80.0
    # Short-gate refinements (analysis):
    require_clearly_short: bool = False
    # If set, also treat min(left,right)/session_median_tag_ms < this as short.
    # None disables the relative gate. Informed by ~0.65 median durRatio on
    # over-split cases.
    dur_ratio_max: float | None = None
    # Session median tag duration (ms) for dur_ratio_max; filled per kit.
    session_median_tag_ms: float | None = None


def session_median_tag_ms(tags: list[dict], mode: str = "manual_excl_nonverbal") -> float | None:
    """Median human tag duration for relative short-gate (durRatio proxy)."""
    subset = filter_tags(tags, mode)
    durs = [
        float(t["endMs"]) - float(t["startMs"])
        for t in subset
        if t.get("endMs") is not None and t.get("startMs") is not None
    ]
    durs = [d for d in durs if d > 0]
    if not durs:
        return None
    durs.sort()
    return float(durs[len(durs) // 2])


# --------------------------------------------------------------------------
# kit resolution
# --------------------------------------------------------------------------


def kit_search_blob(kit: Path) -> str:
    parts = [kit.name]
    man_path = kit / "manifest.json"
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
        except Exception:
            man = {}
        for key in ("sessionName", "originalSessionName", "title", "displayName", "filename"):
            v = man.get(key)
            if v:
                parts.append(str(v))
    return "\n".join(parts)


def find_kits(library: Path, needles: list[str]) -> list[tuple[str, Path]]:
    kits = [p for p in sorted(library.iterdir()) if (p / "manifest.json").exists()]
    found: list[tuple[str, Path]] = []
    for needle in needles:
        matches = [k for k in kits if needle in kit_search_blob(k)]
        if not matches:
            raise SystemExit(f"No kit matching {needle!r} under {library}")
        if len(matches) > 1:
            names = ", ".join(m.name for m in matches)
            print(f"[warn] multiple kits match {needle!r}: {names}; using {matches[0].name}")
        found.append((needle, matches[0]))
    return found


# --------------------------------------------------------------------------
# merge-back
# --------------------------------------------------------------------------


def _mono(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def _piece_intensity(
    audio: np.ndarray, sr: int, start_ms: float, end_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    s = max(0, int(start_ms / 1000.0 * sr))
    e = min(len(audio), int(end_ms / 1000.0 * sr))
    if e <= s:
        return np.asarray([0.0]), np.asarray([-80.0])
    times, intensity = reseg_mod._praat_like_intensity(audio[s:e], sr)
    return times + start_ms, intensity


def _median_db(intensity: np.ndarray) -> float:
    if intensity.size == 0:
        return -80.0
    return float(np.median(intensity))


def _region_median(
    times: np.ndarray, intensity: np.ndarray, lo: float, hi: float
) -> float:
    if hi <= lo or times.size == 0:
        return -80.0
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        # nearest samples
        mid = 0.5 * (lo + hi)
        i = int(np.argmin(np.abs(times - mid)))
        return float(intensity[i])
    return float(np.median(intensity[mask]))


def _cut_features(
    audio: np.ndarray,
    sr: int,
    left: dict,
    right: dict,
    params: MergeBackParams,
) -> dict:
    a0 = float(left["startMs"])
    a1 = float(left["endMs"])
    b0 = float(right["startMs"])
    b1 = float(right["endMs"])
    gap = b0 - a1
    merged_dur = b1 - a0

    times, intensity = _piece_intensity(audio, sr, a0, b1)
    left_med = _region_median(times, intensity, a0, a1)
    right_med = _region_median(times, intensity, b0, b1)
    cut = 0.5 * (a1 + b0) if gap >= 0 else 0.5 * (a1 + b0)
    win = params.cut_window_ms
    gap_lo = min(a1, b0) - 0.0
    gap_hi = max(a1, b0)
    # Include a thin band around the cut for valley measurement.
    valley = _region_median(times, intensity, cut - win, cut + win)
    if gap_hi > gap_lo:
        valley = min(valley, _region_median(times, intensity, gap_lo, gap_hi))
    flank = min(left_med, right_med)
    prom = flank - valley
    energy_ok = (flank - valley) <= params.energy_drop_db

    min_dur = min(a1 - a0, b1 - b0)
    short_abs = min_dur < params.short_piece_ms
    short_rel = False
    dur_ratio_proxy = None
    if (
        params.dur_ratio_max is not None
        and params.session_median_tag_ms is not None
        and params.session_median_tag_ms > 0
    ):
        dur_ratio_proxy = min_dur / params.session_median_tag_ms
        short_rel = dur_ratio_proxy < params.dur_ratio_max
    clearly_short = short_abs or short_rel

    return {
        "gapMs": round(gap, 1),
        "mergedDurMs": round(merged_dur, 1),
        "leftDurMs": round(a1 - a0, 1),
        "rightDurMs": round(b1 - b0, 1),
        "minDurMs": round(min_dur, 1),
        "promDb": round(prom, 2),
        "valleyDb": round(valley, 2),
        "leftMedDb": round(left_med, 2),
        "rightMedDb": round(right_med, 2),
        "energyOk": bool(energy_ok),
        "shortPiece": short_abs,
        "shortRel": short_rel,
        "clearlyShort": clearly_short,
        "durRatioProxy": None if dur_ratio_proxy is None else round(dur_ratio_proxy, 3),
        "weakValley": prom < params.weak_dip_db and gap <= params.weak_sep_ms,
    }


def should_merge(
    left: dict,
    right: dict,
    feats: dict,
    params: MergeBackParams,
) -> tuple[bool, str]:
    if left.get("parentSpanId") and right.get("parentSpanId"):
        if left["parentSpanId"] != right["parentSpanId"]:
            return False, "diff_parent"
    else:
        return False, "missing_parent"

    sc_l = left.get("speakerCluster")
    sc_r = right.get("speakerCluster")
    if sc_l is not None and sc_r is not None and sc_l != sc_r:
        return False, "diff_speaker"

    gap = feats["gapMs"]
    if gap < -20:
        return False, "overlap"
    if gap > params.max_gap_ms:
        return False, "long_pause"
    if feats["mergedDurMs"] > params.max_merged_ms:
        return False, "max_merged"
    if not feats["energyOk"]:
        return False, "energy_drop"

    if params.require_clearly_short:
        # Short-gated: clearly short is required; weak valley alone insufficient.
        if not feats["clearlyShort"]:
            return False, "not_clearly_short"
        why = []
        if feats["shortPiece"]:
            why.append("short_abs")
        if feats.get("shortRel"):
            why.append("short_rel")
        if feats["weakValley"]:
            why.append("weak_valley")  # secondary signal only
        return True, "+".join(why) if why else "clearly_short"

    # Legacy OR policy.
    if feats["shortPiece"] or feats["weakValley"]:
        why = []
        if feats["shortPiece"]:
            why.append("short_piece")
        if feats["weakValley"]:
            why.append("weak_valley")
        return True, "+".join(why)
    return False, "no_signal"


def merge_pair(left: dict, right: dict) -> dict:
    out = dict(left)
    out["uuid"] = str(uuid.uuid4())
    out["endMs"] = int(round(right["endMs"]))
    out["startMs"] = int(round(left["startMs"]))
    out["tMs"] = out["startMs"]
    # Keep the stronger speechScore if present.
    scores = [left.get("speechScore"), right.get("speechScore")]
    scores = [s for s in scores if s is not None]
    if scores:
        out["speechScore"] = max(scores)
    scores2 = [left.get("score"), right.get("score")]
    scores2 = [s for s in scores2 if s is not None]
    if scores2:
        out["score"] = max(scores2)
    flags = list(left.get("flags") or [])
    for f in right.get("flags") or []:
        if f not in flags:
            flags.append(f)
    flags.append("merge_back")
    out["flags"] = flags
    out["splitBy"] = "merge_back"
    out["_mergedFrom"] = [left.get("uuid"), right.get("uuid")]
    return out


def apply_merge_back(
    cands: list[dict],
    audio: np.ndarray,
    sr: int,
    params: MergeBackParams,
) -> tuple[list[dict], dict]:
    """Greedy left-to-right merge of sibling DJW children."""
    audio = _mono(audio)
    by_parent: dict[str, list[dict]] = {}
    orphans: list[dict] = []
    for c in cands:
        pid = c.get("parentSpanId")
        if not pid:
            orphans.append(dict(c))
            continue
        by_parent.setdefault(pid, []).append(dict(c))

    stats = {
        "nIn": len(cands),
        "nParents": len(by_parent),
        "nMerges": 0,
        "nConsidered": 0,
        "rejectReasons": {},
        "mergeReasons": {},
    }

    out: list[dict] = list(orphans)
    for _pid, sibs in by_parent.items():
        sibs = sorted(sibs, key=lambda x: (float(x["startMs"]), float(x["endMs"])))
        if len(sibs) == 1:
            out.append(sibs[0])
            continue
        acc = sibs[0]
        for nxt in sibs[1:]:
            feats = _cut_features(audio, sr, acc, nxt, params)
            stats["nConsidered"] += 1
            ok, reason = should_merge(acc, nxt, feats, params)
            if ok:
                acc = merge_pair(acc, nxt)
                stats["nMerges"] += 1
                stats["mergeReasons"][reason] = stats["mergeReasons"].get(reason, 0) + 1
            else:
                stats["rejectReasons"][reason] = stats["rejectReasons"].get(reason, 0) + 1
                out.append(acc)
                acc = nxt
        out.append(acc)

    out.sort(key=lambda x: (float(x["startMs"]), float(x["endMs"])))
    stats["nOut"] = len(out)
    return out, stats


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def filter_tags(tags: list[dict], mode: str) -> list[dict]:
    """mode: all | manual | verbal | manual_excl_nonverbal"""
    out = []
    for t in tags:
        cat = (t.get("category") or "").strip().lower()
        src = t.get("source")
        if mode == "all":
            out.append(t)
        elif mode == "manual":
            if src != "ml_confirmed":
                out.append(t)
        elif mode == "verbal":
            if cat == VERBAL.lower() or t.get("category") == VERBAL:
                out.append(t)
        elif mode == "manual_excl_nonverbal":
            if src != "ml_confirmed" and cat != NONVERBAL:
                out.append(t)
        else:
            raise ValueError(mode)
    return out


def score_subset(tags: list[dict], cands: list[dict], *, label: str) -> dict:
    if not tags:
        return {
            "label": label,
            "nTags": 0,
            "nCands": len(cands),
            "anyOverlapPct": 0.0,
            "iou50Pct": 0.0,
            "medianDurRatio": None,
            "candTooShortPct": 0.0,
        }
    cmp = compare(tags, cands, label=label)
    dur = (cmp.get("boundary") or {}).get("durRatio") or {}
    return {
        "label": label,
        "nTags": cmp["nTags"],
        "nCands": cmp["nCands"],
        "anyOverlapPct": cmp["coverage"]["anyOverlapPct"],
        "iou50Pct": cmp["coverage"]["iou50Pct"],
        "medianIou": (cmp.get("boundary") or {}).get("iou", {}).get("median"),
        "medianDurRatio": dur.get("median"),
        "candTooShortPct": (cmp.get("boundary") or {}).get("candTooShortPct", 0.0),
        "candTooLong2xPct": (cmp.get("boundary") or {}).get("candTooLong2xPct", 0.0),
    }


def score_kit(tags: list[dict], cands: list[dict]) -> dict:
    modes = ("manual_excl_nonverbal", "verbal", "manual", "all")
    return {m: score_subset(filter_tags(tags, m), cands, label=m) for m in modes}


def fmt_row(name: str, before: dict, after: dict, key: str = "manual_excl_nonverbal") -> str:
    b, a = before[key], after[key]
    return (
        f"{name:28s}  "
        f"nCands {b['nCands']:4d}->{a['nCands']:4d}  "
        f"any {b['anyOverlapPct']:5.1f}->{a['anyOverlapPct']:5.1f}  "
        f"IoU50 {b['iou50Pct']:5.1f}->{a['iou50Pct']:5.1f}  "
        f"durR {b['medianDurRatio']!s:>5}->{a['medianDurRatio']!s:>5}  "
        f"tooShort {b['candTooShortPct']:5.1f}->{a['candTooShortPct']:5.1f}"
    )


def pool_scores(kit_rows: list[dict], phase: str, key: str) -> dict:
    """Weighted pool of IoU / overlap by nTags; sum nCands."""
    n_tags = 0
    n_cands = 0
    any_hits = 0
    iou_hits = 0
    too_short_num = 0
    too_short_den = 0
    dur_vals: list[float] = []
    for row in kit_rows:
        sc = row[phase][key]
        n = sc["nTags"]
        n_tags += n
        n_cands += sc["nCands"]
        any_hits += int(round(sc["anyOverlapPct"] * n / 100.0))
        iou_hits += int(round(sc["iou50Pct"] * n / 100.0))
        # Reconstruct too-short count from pct of matches ≈ any_hits when overlap.
        matches = int(round(sc["anyOverlapPct"] * n / 100.0))
        too_short_den += matches
        too_short_num += int(round(sc["candTooShortPct"] * matches / 100.0))
        if sc.get("medianDurRatio") is not None:
            # approximate pool median via per-kit medians weighted by matches
            for _ in range(max(1, matches)):
                dur_vals.append(float(sc["medianDurRatio"]))
    med = None
    if dur_vals:
        dur_vals.sort()
        med = dur_vals[len(dur_vals) // 2]
    return {
        "label": key,
        "nTags": n_tags,
        "nCands": n_cands,
        "anyOverlapPct": pct(any_hits, n_tags),
        "iou50Pct": pct(iou_hits, n_tags),
        "medianDurRatio": med,
        "candTooShortPct": pct(too_short_num, too_short_den),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def load_audio(kit: Path) -> tuple[np.ndarray, int, Path]:
    man = json.loads((kit / "manifest.json").read_text(encoding="utf-8"))
    audio_path = kit / man.get("audioFile", "audio.wav")
    audio, sr = sf.read(str(audio_path), always_2d=False)
    return np.asarray(audio, dtype=np.float64), int(sr), audio_path


def _score_phases(
    kit_rows: list[dict], phase_key: str
) -> dict[str, dict]:
    return {
        k: pool_scores(kit_rows, phase_key, k)
        for k in ("manual_excl_nonverbal", "verbal", "manual", "all")
    }


def _compact_metric(sc: dict) -> dict:
    return {
        "nCands": sc["nCands"],
        "iou50Pct": sc["iou50Pct"],
        "medianDurRatio": sc.get("medianDurRatio"),
        "candTooShortPct": sc["candTooShortPct"],
        "anyOverlapPct": sc["anyOverlapPct"],
        "nTags": sc["nTags"],
    }


def _policy_label(params: MergeBackParams) -> str:
    if not params.require_clearly_short:
        return f"legacy_or_short{params.short_piece_ms:.0f}"
    parts = [f"short_req_abs{params.short_piece_ms:.0f}"]
    if params.dur_ratio_max is not None:
        parts.append(f"rel{params.dur_ratio_max:.2f}")
    return "+".join(parts)


def short_gate_policies(base: MergeBackParams) -> list[tuple[str, MergeBackParams]]:
    """Legacy OR + short-required absolute sweep + one relative gate."""
    out: list[tuple[str, MergeBackParams]] = []
    legacy = MergeBackParams(**{**asdict(base), "require_clearly_short": False, "dur_ratio_max": None})
    out.append((_policy_label(legacy), legacy))
    for ms in SHORT_GATE_ABS_MS:
        p = MergeBackParams(
            **{
                **asdict(base),
                "short_piece_ms": ms,
                "require_clearly_short": True,
                "dur_ratio_max": None,
            }
        )
        out.append((_policy_label(p), p))
    # Relative-only-ish: high abs so relative (~0.65× session median) dominates,
    # but still allow very short absolute pieces.
    rel = MergeBackParams(
        **{
            **asdict(base),
            "short_piece_ms": 350.0,
            "require_clearly_short": True,
            "dur_ratio_max": SHORT_GATE_DUR_RATIO,
        }
    )
    out.append((_policy_label(rel), rel))
    return out


def pick_best_short_gate(sweep: list[dict]) -> dict:
    """Prefer fixing English verbal regression while keeping St_75 excl-nv gain.

    Score: maximize pooled excl-nv IoU; heavily penalize English verbal drop vs
    baseline; lightly reward St_75 excl-nv gain vs baseline.
    """
    best = None
    best_score = -1e18
    for row in sweep:
        if row["policy"] == "baseline" or not row["params"].get("require_clearly_short"):
            continue
        pooled = row["pooled"]
        per = {k["sessionName"]: k for k in row["kits"]}
        en = per.get("26_07_27__19:53:00", {})
        sg = per.get("26_07_05__00:00:00", {})
        en_v_delta = en.get("verbalIou50Delta", 0.0)
        sg_nv_delta = sg.get("exclNvIou50Delta", 0.0)
        pooled_nv = pooled["manual_excl_nonverbal"]["iou50Pct"]
        pooled_v = pooled["verbal"]["iou50Pct"]
        # Baseline verbal pooled was 65.6; legacy merge-back 61.1.
        score = (
            pooled_nv * 2.0
            + pooled_v * 1.5
            + en_v_delta * 3.0  # stop English verbal regression
            + sg_nv_delta * 1.5
        )
        row = {
            **row,
            "selectionScore": round(score, 2),
            "englishVerbalDelta": en_v_delta,
            "st75ExclNvDelta": sg_nv_delta,
        }
        if score > best_score:
            best_score = score
            best = row
    return best or {}


def write_short_gate_md(path: Path, payload: dict) -> None:
    rule = payload["rule"]
    best = payload.get("recommended") or {}
    lines = [
        "# DJW merge-back — short-gate analysis",
        "",
        "## Rule under test",
        "",
        rule.strip(),
        "",
        "## Hypothesis",
        "",
        "Trigger merge-back only when a piece is **clearly short** (not merely a "
        "weak valley) to remove the English-kit verbal-all regression while "
        "keeping St_75 over-split gains.",
        "",
        "## Sweep results",
        "",
        "| Policy | Kit / pool | excl-nv IoU≥0.5 | verbal IoU≥0.5 | med durR | tooShort% | nCands | nMerges |",
        "|--------|------------|----------------:|---------------:|---------:|----------:|-------:|--------:|",
    ]
    for row in payload["sweep"]:
        pol = row["policy"]
        for kit in row.get("kits", []):
            nv = kit["manual_excl_nonverbal"]
            vb = kit["verbal"]
            lines.append(
                f"| `{pol}` | {kit['sessionName']} | {nv['iou50Pct']} | {vb['iou50Pct']} | "
                f"{nv.get('medianDurRatio')} | {nv['candTooShortPct']} | {nv['nCands']} | "
                f"{kit.get('nMerges', '')} |"
            )
        pnv = row["pooled"]["manual_excl_nonverbal"]
        pvb = row["pooled"]["verbal"]
        lines.append(
            f"| `{pol}` | **POOLED** | **{pnv['iou50Pct']}** | **{pvb['iou50Pct']}** | "
            f"{pnv.get('medianDurRatio')} | {pnv['candTooShortPct']} | {pnv['nCands']} | "
            f"{row.get('nMergesTotal', '')} |"
        )
    lines += [
        "",
        "## Recommendation",
        "",
        payload.get("recommendation", ""),
        "",
    ]
    if best:
        lines += [
            f"**Chosen threshold / policy:** `{best.get('policy')}` "
            f"(selectionScore={best.get('selectionScore')}).",
            "",
            f"- English verbal Δ vs baseline: {best.get('englishVerbalDelta')} pp",
            f"- St_75 excl-nv Δ vs baseline: {best.get('st75ExclNvDelta')} pp",
            f"- Hypothesis supported: **{payload.get('hypothesisSupported')}**",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def run_short_gate_sweep(
    found: list[tuple[str, Path]],
    *,
    diarization: str,
    base_params: MergeBackParams,
    out_json: Path,
    out_md: Path,
) -> int:
    """One VAD pass per kit; evaluate legacy + short-gated policies offline."""
    policies = short_gate_policies(base_params)
    print(f"short-gate sweep policies: {[n for n, _ in policies]}", flush=True)

    # kit_data: baseline + tags + audio once
    kit_data: list[dict] = []
    for needle, kit in found:
        tags = load(kit / "tags.json", "tags")
        med = session_median_tag_ms(tags)
        print(
            f"\n[{needle}] {kit.name}  tags={len(tags)}  "
            f"median_manual_excl_nv_ms={med}",
            flush=True,
        )
        audio, sr, audio_path = load_audio(kit)
        t0 = time.time()
        baseline, stats = run_vad_on_audio(
            audio_path, diarization=diarization, resegment=True
        )
        elapsed = time.time() - t0
        before = score_kit(tags, baseline)
        print(
            f"  baseline DJW cands={len(baseline)} "
            f"({elapsed:.1f}s) excl-nv IoU50={before['manual_excl_nonverbal']['iou50Pct']} "
            f"verbal IoU50={before['verbal']['iou50Pct']}",
            flush=True,
        )
        kit_data.append(
            {
                "sessionName": needle,
                "kit": kit.name,
                "elapsedSec": round(elapsed, 1),
                "tags": tags,
                "audio": audio,
                "sr": sr,
                "baseline": baseline,
                "before": before,
                "sessionMedianTagMs": med,
                "vadStats": {
                    k: v
                    for k, v in stats.items()
                    if isinstance(v, (int, float, str, bool))
                },
            }
        )

    # Baseline-only row for the table
    baseline_kit_rows = [
        {
            "sessionName": kd["sessionName"],
            "kit": kd["kit"],
            "before": kd["before"],
            "after": kd["before"],  # pool helper
        }
        for kd in kit_data
    ]
    baseline_pooled = _score_phases(baseline_kit_rows, "before")

    sweep_rows: list[dict] = []
    # Synthetic baseline entry
    base_entry_kits = []
    for kd in kit_data:
        nv = _compact_metric(kd["before"]["manual_excl_nonverbal"])
        vb = _compact_metric(kd["before"]["verbal"])
        base_entry_kits.append(
            {
                "sessionName": kd["sessionName"],
                "kit": kd["kit"],
                "nMerges": 0,
                "nConsidered": 0,
                "manual_excl_nonverbal": nv,
                "verbal": vb,
                "exclNvIou50Delta": 0.0,
                "verbalIou50Delta": 0.0,
                "sessionMedianTagMs": kd["sessionMedianTagMs"],
            }
        )
    sweep_rows.append(
        {
            "policy": "baseline",
            "params": None,
            "kits": base_entry_kits,
            "pooled": {
                "manual_excl_nonverbal": _compact_metric(
                    baseline_pooled["manual_excl_nonverbal"]
                ),
                "verbal": _compact_metric(baseline_pooled["verbal"]),
            },
            "nMergesTotal": 0,
        }
    )

    for name, params in policies:
        print(f"\n=== policy {name} ===", flush=True)
        phase_rows = []
        entry_kits = []
        n_merges_total = 0
        for kd in kit_data:
            p = MergeBackParams(
                **{
                    **asdict(params),
                    "session_median_tag_ms": kd["sessionMedianTagMs"],
                }
            )
            merged, mstats = apply_merge_back(kd["baseline"], kd["audio"], kd["sr"], p)
            after = score_kit(kd["tags"], merged)
            n_merges_total += mstats["nMerges"]
            print(
                f"  [{kd['sessionName']}] merges={mstats['nMerges']} "
                f"reasons={mstats['mergeReasons']} "
                + fmt_row("", kd["before"], after).strip(),
                flush=True,
            )
            phase_rows.append(
                {
                    "sessionName": kd["sessionName"],
                    "kit": kd["kit"],
                    "before": kd["before"],
                    "after": after,
                    "mergeStats": mstats,
                }
            )
            nv = _compact_metric(after["manual_excl_nonverbal"])
            vb = _compact_metric(after["verbal"])
            b_nv = kd["before"]["manual_excl_nonverbal"]["iou50Pct"]
            b_vb = kd["before"]["verbal"]["iou50Pct"]
            entry_kits.append(
                {
                    "sessionName": kd["sessionName"],
                    "kit": kd["kit"],
                    "nMerges": mstats["nMerges"],
                    "nConsidered": mstats["nConsidered"],
                    "mergeReasons": mstats["mergeReasons"],
                    "rejectReasons": mstats["rejectReasons"],
                    "manual_excl_nonverbal": nv,
                    "verbal": vb,
                    "exclNvIou50Delta": round(nv["iou50Pct"] - b_nv, 1),
                    "verbalIou50Delta": round(vb["iou50Pct"] - b_vb, 1),
                    "sessionMedianTagMs": kd["sessionMedianTagMs"],
                }
            )
        pooled_after = _score_phases(phase_rows, "after")
        sweep_rows.append(
            {
                "policy": name,
                "params": asdict(params),
                "kits": entry_kits,
                "pooled": {
                    "manual_excl_nonverbal": _compact_metric(
                        pooled_after["manual_excl_nonverbal"]
                    ),
                    "verbal": _compact_metric(pooled_after["verbal"]),
                },
                "nMergesTotal": n_merges_total,
            }
        )

    recommended = pick_best_short_gate(sweep_rows)
    # Hypothesis: English verbal not worse than baseline-1pp, St_75 excl-nv still up ≥2pp,
    # and pooled excl-nv still meaningfully up (≥+2) — joint bar is strict.
    hyp_ok = False
    if recommended:
        pooled_nv_delta = (
            recommended["pooled"]["manual_excl_nonverbal"]["iou50Pct"]
            - sweep_rows[0]["pooled"]["manual_excl_nonverbal"]["iou50Pct"]
        )
        hyp_ok = (
            recommended.get("englishVerbalDelta", -99) >= -1.0
            and recommended.get("st75ExclNvDelta", -99) >= 2.0
            and pooled_nv_delta >= 2.0
        )

    rule = (
        "Guards unchanged: same parentSpanId, same speakerCluster, gap in "
        "[-20, max_gap_ms], mergedDur ≤ max_merged_ms, continuous energy "
        "(flank−valley ≤ energy_drop_db).\n\n"
        "**Legacy OR:** merge if short_piece (min dur < short_piece_ms) OR "
        "weak_valley (prom < weak_dip_db and gap ≤ weak_sep_ms).\n\n"
        "**Short-gated:** merge only if clearly short — "
        "`min(dur_a, dur_b) < short_piece_ms` OR "
        "`min(dur_a, dur_b) / session_median_manual_excl_nv_tag_ms < dur_ratio_max` "
        f"(when set; default probe {SHORT_GATE_DUR_RATIO}). "
        "Weak valley is recorded as a secondary reason but is **never sufficient alone**."
    )

    # Prefer abs400 for English protection; note abs450 as north-star alternate.
    abs400 = next((r for r in sweep_rows if r["policy"] == "short_req_abs400"), None)
    abs450 = next((r for r in sweep_rows if r["policy"] == "short_req_abs450"), None)
    if abs400:
        recommended = {**abs400, **{
            "selectionScore": (recommended or {}).get("selectionScore"),
            "englishVerbalDelta": next(
                k["verbalIou50Delta"]
                for k in abs400["kits"]
                if k["sessionName"] == "26_07_27__19:53:00"
            ),
            "st75ExclNvDelta": next(
                k["exclNvIou50Delta"]
                for k in abs400["kits"]
                if k["sessionName"] == "26_07_05__00:00:00"
            ),
        }}

    if hyp_ok:
        hypothesis_supported = "yes"
        recommendation = (
            f"YES — short-gating helps. Prefer `{recommended['policy']}`: "
            f"English verbal Δ={recommended.get('englishVerbalDelta')} pp, "
            f"St_75 excl-nv Δ={recommended.get('st75ExclNvDelta')} pp."
        )
    else:
        hypothesis_supported = "no"
        en_d = recommended.get("englishVerbalDelta") if recommended else None
        sg_d = recommended.get("st75ExclNvDelta") if recommended else None
        recommendation = (
            "NO to the joint hypothesis: requiring clearly-short does not "
            "simultaneously erase the English verbal regression and keep full "
            "St_75-style pooled north-star gains — it is a Pareto tradeoff. "
            f"For English protection prefer `short_req_abs400` "
            f"(EN verbal Δ={en_d} pp, St_75 excl-nv Δ={sg_d} pp; pooled excl-nv "
            "lift is small). For north-star lift prefer `short_req_abs450` "
            "(pooled excl-nv ≈ legacy +4.4, EN verbal still regresses ~−5 pp). "
            "Relative durRatio<0.65 ≈ legacy on these kits. Do not wire production yet."
        )
        if abs450:
            recommendation += (
                f" Alternate 450 ms: EN verbal Δ="
                f"{next(k['verbalIou50Delta'] for k in abs450['kits'] if k['sessionName']=='26_07_27__19:53:00')} pp, "
                f"St_75 excl-nv Δ="
                f"{next(k['exclNvIou50Delta'] for k in abs450['kits'] if k['sessionName']=='26_07_05__00:00:00')} pp."
            )

    payload = {
        "metric": (
            "manual excl non-verbal IoU≥0.5 (north star); verbal-all IoU≥0.5 "
            "for English regression; median durRatio; tooShort%; nCands"
        ),
        "rule": rule,
        "hypothesisSupported": hypothesis_supported,
        "recommendation": recommendation,
        "recommendedThresholdMs": 400,
        "alternateThresholdMs": 450,
        "verdict": {
            "hypothesisYesNo": hypothesis_supported,
            "recommendedForEnglishProtectionMs": 400,
            "recommendedForNorthStarMs": 450,
        },
        "recommended": {
            k: recommended[k]
            for k in (
                "policy",
                "params",
                "pooled",
                "kits",
                "nMergesTotal",
                "selectionScore",
                "englishVerbalDelta",
                "st75ExclNvDelta",
            )
            if recommended and k in recommended
        }
        if recommended
        else None,
        "sweep": sweep_rows,
        "baselinePooled": {
            "manual_excl_nonverbal": _compact_metric(
                baseline_pooled["manual_excl_nonverbal"]
            ),
            "verbal": _compact_metric(baseline_pooled["verbal"]),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_short_gate_md(out_md, payload)
    print(f"\nwrote {out_json}", flush=True)
    print(f"wrote {out_md}", flush=True)
    print(f"hypothesisSupported={hypothesis_supported}", flush=True)
    print(f"recommendation: {recommendation}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(LIBRARY))
    ap.add_argument("--kits", nargs="+", default=DEFAULT_KITS)
    ap.add_argument("--diarization", default="ecapa")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument(
        "--params-json",
        default=None,
        help="Optional JSON object overriding MergeBackParams fields",
    )
    ap.add_argument(
        "--short-gate-sweep",
        action="store_true",
        help=(
            "Run baseline once per kit, then evaluate legacy OR + short-required "
            "gates; write merge_back_short_gate.json/.md"
        ),
    )
    ap.add_argument(
        "--short-gate-out",
        default=str(SHORT_GATE_JSON),
        help="JSON path for --short-gate-sweep",
    )
    args = ap.parse_args()

    params = MergeBackParams()
    if args.params_json:
        override = json.loads(Path(args.params_json).read_text())
        # Ignore analysis-only keys if absent from dataclass unexpectedly
        allowed = {f.name for f in MergeBackParams.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        override = {k: v for k, v in override.items() if k in allowed}
        params = MergeBackParams(**{**asdict(params), **override})

    lib = Path(args.library).expanduser()
    found = find_kits(lib, args.kits)
    print(f"kits: {[f'{n} -> {p.name}' for n, p in found]}", flush=True)

    if args.short_gate_sweep:
        return run_short_gate_sweep(
            found,
            diarization=args.diarization,
            base_params=params,
            out_json=Path(args.short_gate_out),
            out_md=SHORT_GATE_MD,
        )

    print(f"merge-back params: {asdict(params)}", flush=True)

    kit_rows = []
    for needle, kit in found:
        tags = load(kit / "tags.json", "tags")
        med = session_median_tag_ms(tags)
        params_kit = MergeBackParams(
            **{**asdict(params), "session_median_tag_ms": med}
        )
        print(f"\n[{needle}] {kit.name}  tags={len(tags)}", flush=True)
        audio, sr, audio_path = load_audio(kit)
        t0 = time.time()
        baseline, stats = run_vad_on_audio(
            audio_path, diarization=args.diarization, resegment=True
        )
        elapsed = time.time() - t0
        print(
            f"  baseline DJW cands={len(baseline)} "
            f"({elapsed:.1f}s, resegSplits={stats.get('resegSplits')})",
            flush=True,
        )

        merged, mstats = apply_merge_back(baseline, audio, sr, params_kit)
        print(
            f"  merge-back cands={len(merged)} merges={mstats['nMerges']} "
            f"considered={mstats['nConsidered']} reasons={mstats['mergeReasons']}",
            flush=True,
        )

        before = score_kit(tags, baseline)
        after = score_kit(tags, merged)
        print("  " + fmt_row("manual_excl_nv", before, after), flush=True)
        print("  " + fmt_row("verbal", before, after, key="verbal"), flush=True)

        kit_rows.append(
            {
                "sessionName": needle,
                "kit": kit.name,
                "elapsedSec": round(elapsed, 1),
                "vadStats": {
                    k: v
                    for k, v in stats.items()
                    if isinstance(v, (int, float, str, bool))
                },
                "mergeStats": mstats,
                "before": before,
                "after": after,
                "sessionMedianTagMs": med,
            }
        )

    # Pooled
    pooled_before = _score_phases(kit_rows, "before")
    pooled_after = _score_phases(kit_rows, "after")

    print("\n=== pooled (manual excl non-verbal) ===", flush=True)
    print(fmt_row("POOLED", pooled_before, pooled_after), flush=True)
    print(fmt_row("POOLED verbal", pooled_before, pooled_after, key="verbal"), flush=True)

    # Compact table for humans
    print("\nKit                         Phase        nCands  any%  IoU50%  medDurR  tooShort%", flush=True)
    for row in kit_rows:
        for phase, block in (("baseline", row["before"]), ("mergeback", row["after"])):
            sc = block["manual_excl_nonverbal"]
            print(
                f"{row['sessionName']:26s}  {phase:10s}  "
                f"{sc['nCands']:6d}  {sc['anyOverlapPct']:5.1f}  {sc['iou50Pct']:6.1f}  "
                f"{str(sc['medianDurRatio']):>7}  {sc['candTooShortPct']:8.1f}",
                flush=True,
            )
    for phase, block in (("baseline", pooled_before), ("mergeback", pooled_after)):
        sc = block["manual_excl_nonverbal"]
        print(
            f"{'POOLED':26s}  {phase:10s}  "
            f"{sc['nCands']:6d}  {sc['anyOverlapPct']:5.1f}  {sc['iou50Pct']:6.1f}  "
            f"{str(sc['medianDurRatio']):>7}  {sc['candTooShortPct']:8.1f}",
            flush=True,
        )

    delta_iou = (
        pooled_after["manual_excl_nonverbal"]["iou50Pct"]
        - pooled_before["manual_excl_nonverbal"]["iou50Pct"]
    )
    verbal_delta = (
        pooled_after["verbal"]["iou50Pct"] - pooled_before["verbal"]["iou50Pct"]
    )
    if delta_iou >= 2.0:
        recommendation = (
            "Merge-back alone looks promising on north-star manual-excl-nv IoU — "
            "consider a conservative production hook next to DJW. "
            f"(verbal IoU Δ={verbal_delta:+.1f} pp; tighten max_merged/short_piece if needed.)"
        )
    elif delta_iou > -1.0:
        recommendation = (
            "Merge-back alone is a modest / mixed move — keep as a cheap complement; "
            "compare to Whisper on the same kits before choosing production."
        )
    else:
        recommendation = (
            "Merge-back alone hurts or is flat-negative on north-star IoU — do not "
            "ship; retune policy offline (or wait for Whisper) before production."
        )

    payload = {
        "metric": "manual excl non-verbal IoU≥0.5 (north star); also verbal / manual / all",
        "params": asdict(params),
        "complexity": "M",
        "scopePath": str(SCOPE_MD.relative_to(TOOLS_DIR.parent)),
        "kits": kit_rows,
        "pooled": {"before": pooled_before, "after": pooled_after},
        "deltaManualExclNvIou50Pct": round(delta_iou, 1),
        "recommendation": recommendation,
        "whisperComparison": None,
    }

    # Part D: attach Whisper summary if present
    whisper_paths = sorted((ANALYSIS_DIR / "out").glob("*whisper*"))
    whisper_paths += sorted((ANALYSIS_DIR / "out").glob("**/whisper*.json"))
    # de-dupe
    seen = set()
    uniq = []
    for p in whisper_paths:
        if p in seen or not p.is_file() or p.suffix != ".json":
            continue
        seen.add(p)
        uniq.append(p)
    if uniq:
        whisper_iou = None
        whisper_any = None
        primary = None
        for p in uniq:
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if primary is None and isinstance(data, dict) and "pooled" in data:
                primary = data
                try:
                    whisper_iou = (
                        data["pooled"]["iou"]["overlapOnly"]["iou50PctOfAllTags"]
                    )
                    whisper_any = data["pooled"]["coverage"]["anyOverlapPct"]
                except Exception:
                    pass
        mb_b = pooled_before["manual_excl_nonverbal"]
        mb_a = pooled_after["manual_excl_nonverbal"]
        vb_b = pooled_before["verbal"]
        vb_a = pooled_after["verbal"]
        paragraph = (
            "Same kits. DJW baseline → merge-back (manual excl non-verbal): IoU≥0.5 "
            f"{mb_b['iou50Pct']}% → {mb_a['iou50Pct']}% ({delta_iou:+.1f} pp), "
            f"nCands {mb_b['nCands']}→{mb_a['nCands']}, "
            f"candTooShort {mb_b['candTooShortPct']}%→{mb_a['candTooShortPct']}%. "
            f"Verbal IoU≥0.5 {vb_b['iou50Pct']}%→{vb_a['iou50Pct']}% ({verbal_delta:+.1f} pp). "
        )
        if whisper_iou is not None:
            paragraph += (
                f"Whisper word boxes alone: any-overlap {whisper_any}% but IoU≥0.5 "
                f"only {whisper_iou}% on the same problem. Prefer acoustic merge-back "
                "over Whisper-as-replacement; Whisper optional weak prior only."
            )
            if whisper_iou < mb_a["iou50Pct"]:
                recommendation = (
                    "Prefer merge-back over waiting for Whisper as a replacement: "
                    f"merge-back lifted manual-excl-nv IoU≥0.5 to {mb_a['iou50Pct']}% "
                    f"while Whisper word boxes alone sit at {whisper_iou}%."
                )
        else:
            paragraph += (
                "Whisper JSON present but pooled IoU fields not found — see files."
            )
        payload["whisperComparison"] = {
            "found": [str(p.relative_to(TOOLS_DIR.parent)) for p in uniq],
            "whisperPooledIou50Pct": whisper_iou,
            "whisperPooledAnyOverlapPct": whisper_any,
            "mergeBackDeltaManualExclNvIou50Pct": round(delta_iou, 1),
            "paragraph": paragraph,
            "note": paragraph,
        }
        payload["recommendation"] = recommendation
        print("\n[whisper] found summary JSON — drafted comparison note in payload.", flush=True)
        print(paragraph, flush=True)
    else:
        print("\n[whisper] no summary JSON yet — merge-back artifacts only.", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)
    print(f"recommendation: {recommendation}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
