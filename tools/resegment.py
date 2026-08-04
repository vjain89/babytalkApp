"""Syllable-nucleus resegmentation + short-gated merge-back for ML candidates.

Takes a speaker-homogeneous (or VAD) span that may hold several words or
syllables and cuts it into **word-like** (tag-sized) children for Review.

This is stage 3b–3c of the ML-candidates pipeline:

    stage 1  VAD + speechlike
    stage 2  diarization (optional)
    stage 3a pause-split spans still over ~4s
    stage 3b **this module** — de Jong & Wempe nuclei → children
    stage 3c **merge-back** — glue clearly-short over-splits into word-like spans
    stage 3d per-piece speech gate → annotations.json

Parents (longer VAD/diarization regions) stay pipeline-internal via
``parentSpanId``. What Review should see are word-like candidates after DJW
**plus** short-gated merge-back — not raw syllable shards.

Algorithm follows de Jong & Wempe (2009), *Praat script to detect syllable
nuclei and measure speech rate automatically* (Behavior Research Methods),
adapted to *segment* (not only count) and tuned for baby/caregiver home audio:

  1. Intensity envelope with Praat-like smoothing
     (``To Intensity… 50`` ⇒ ~64 ms averaging window).
  2. Silence / ignorance floor = median intensity + ``IGNORANCE_DB``
     (0 dB for unfiltered sounds, per the original script notes).
  3. Candidate peaks = local intensity maxima above that floor.
  4. Keep a peak only if the **preceding dip** to the next peak is
     ≥ ``MIN_DIP_DB`` (2 dB unfiltered default in DJW).
  5. Discard peaks that are not **voiced** (autocorr pitch present).
  6. Cut the parent at intensity minima between consecutive kept nuclei.
  7. BabyTalk extras: trim each child to above-threshold intensity; force-split
     leftovers still longer than ``RESEG_TARGET_MS``.
  8. Merge-back: reunite clearly-short sibling pieces across weak cuts
     (``require_clearly_short`` + ``short_piece_ms=400``, ``max_gap_ms=200`` —
     production pair; looser 450/300 remains not accepted; weak valley is
     secondary only).

Whisper word timestamps are intentionally *not* used as the word-box source
(ruled out — see ``docs/IOU_WHISPER_VS_MERGEBACK.md``).

Citation:
  De Jong, N. H., & Wempe, T. (2009). Praat script to detect syllable nuclei
  and measure speech rate automatically. Behavior Research Methods, 41(2),
  385–390. https://doi.org/10.3758/BRM.41.2.385
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Iterable

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install deps: pip install numpy\n" + str(e)) from e

# --- BabyTalk review targets (not part of DJW) -----------------------------
# Soft upper bound after DJW nuclei cuts. Only force-splits leftovers.
RESEG_TARGET_MS = 1_600.0
RESEG_MIN_PART_MS = 420.0
RESEG_MIN_PARENT_MS = 700.0
VOICE_TRIM_PAD_MS = 40.0

# --- de Jong & Wempe / Praat defaults --------------------------------------
# Praat "To Intensity... 50 0 yes" → averaging window ≈ 3.2 / min_pitch s.
INTENSITY_MIN_PITCH_HZ = 50.0
INTENSITY_WINDOW_S = 3.2 / INTENSITY_MIN_PITCH_HZ
HOP_MS = 10.0
# Original script: Ignorance Level = 0 (unfiltered) or 2 (filtered).
IGNORANCE_DB = 0.0
# Original script: Minimum dip between peaks = 2 (unfiltered) or 4 (filtered).
MIN_DIP_DB = 2.0
# Voicing gate for nuclei. Slightly below speechlike's 0.42 for breathy toddlers.
VOICING_PEAK = 0.32
F0_MIN_HZ = 70.0
F0_MAX_HZ = 600.0
# Force-split uses the same dip depth as DJW (do not invent weaker cuts).
FORCE_SPLIT_MIN_DIP_DB = 2.0
# After DJW finds nuclei, only *cut* between nuclei that look like word
# boundaries — close nuclei with a shallow valley are syllables of one word
# (e.g. änteli) and should stay in one candidate tag.
WORD_SPLIT_MIN_SEP_MS = 300.0
WORD_SPLIT_MIN_DIP_DB = 4.0

# --- Short-gated merge-back (post-DJW) --------------------------------------
# Production defaults: short_piece_ms=400, max_gap_ms=200 (require_clearly_short).
# Active-audio dismiss-vs-tag accepted 100→200 (2026-08-03). Looser 450/300
# from raw oracle gaps remains not accepted. Weak valley alone is never enough.
# See tools/analysis/out/dismiss_vs_tag_delta_26_07_27_active.md;
# docs/DECISIONS.md.
MERGE_BACK_MAX_GAP_MS = 200.0
MERGE_BACK_MAX_MERGED_MS = 1800.0
MERGE_BACK_SHORT_PIECE_MS = 400.0
MERGE_BACK_WEAK_DIP_DB = 5.0
MERGE_BACK_WEAK_SEP_MS = 280.0
MERGE_BACK_ENERGY_DROP_DB = 6.0
MERGE_BACK_CUT_WINDOW_MS = 80.0


@dataclass(frozen=True)
class MergeBackParams:
    """Post-DJW merge-back policy knobs.

    Short-gated policy (production default, ``require_clearly_short=True``):
    weak valley alone is **not** enough. Merge only when a piece is clearly
    short by absolute ms and/or vs session tag-duration median
    (``dur_ratio_max``), then treat weak-valley as a secondary label only.

    Legacy OR policy (``require_clearly_short=False``): merge when
    ``short_piece OR weak_valley`` (plus guards) — used by analysis sweeps.
    """

    max_gap_ms: float = MERGE_BACK_MAX_GAP_MS
    max_merged_ms: float = MERGE_BACK_MAX_MERGED_MS
    short_piece_ms: float = MERGE_BACK_SHORT_PIECE_MS
    weak_dip_db: float = MERGE_BACK_WEAK_DIP_DB
    weak_sep_ms: float = MERGE_BACK_WEAK_SEP_MS
    energy_drop_db: float = MERGE_BACK_ENERGY_DROP_DB
    cut_window_ms: float = MERGE_BACK_CUT_WINDOW_MS
    require_clearly_short: bool = True
    # If set, also treat min(left,right)/session_median_tag_ms < this as short.
    dur_ratio_max: float | None = None
    session_median_tag_ms: float | None = None


def _mono(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def _praat_like_intensity(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(times_ms, intensity_db) approximating Praat Intensity (min pitch 50)."""
    x = _mono(x)
    hop = max(1, int(round(sr * HOP_MS / 1000.0)))
    win = max(hop * 2, int(round(sr * INTENSITY_WINDOW_S)))
    if win % 2 == 0:
        win += 1
    if len(x) < 8:
        rms = float(np.sqrt(np.mean(x * x)) + 1e-12) if len(x) else 1e-12
        return np.asarray([0.0]), np.asarray([20.0 * math.log10(rms)])

    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=np.float64) / win
    ms = np.convolve(xp * xp, kernel, mode="valid")
    ms = ms[::hop]
    times = (np.arange(len(ms)) * hop) * 1000.0 / sr
    intensity = 10.0 * np.log10(ms + 1e-12)

    sigma_frames = max(1.0, (INTENSITY_WINDOW_S * 1000.0 / HOP_MS) / 6.0)
    radius = int(max(1, round(3 * sigma_frames)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-0.5 * (t / sigma_frames) ** 2)
    g /= g.sum()
    intensity = np.convolve(intensity, g, mode="same")
    return times, intensity


def _voiced_at_peaks(
    x: np.ndarray, sr: int, times_ms: np.ndarray, peak_indices: list[int]
) -> list[bool]:
    """Whether each peak time has a voiced frame (DJW voicedness check)."""
    x = _mono(x)
    frame = max(16, int(sr * 0.040))
    min_lag = max(2, int(sr / F0_MAX_HZ))
    max_lag = min(frame - 1, int(sr / F0_MIN_HZ))
    if max_lag <= min_lag:
        return [True] * len(peak_indices)

    nfft = 1
    while nfft < 2 * frame:
        nfft *= 2
    out: list[bool] = []
    for pi in peak_indices:
        t_ms = float(times_ms[pi])
        center = int(round(t_ms / 1000.0 * sr))
        s = max(0, center - frame // 2)
        e = min(len(x), s + frame)
        chunk = x[s:e]
        if len(chunk) < frame:
            pad = np.zeros(frame)
            pad[: len(chunk)] = chunk
            chunk = pad
        f = chunk - chunk.mean()
        spec = np.fft.rfft(f, n=nfft)
        acf = np.fft.irfft(spec * np.conj(spec), n=nfft)[:frame]
        z = acf[0] if acf[0] > 0 else 1e-12
        acf = acf / z
        peak = float(np.max(acf[min_lag : max_lag + 1]))
        out.append(peak >= VOICING_PEAK)
    return out


def _local_maxima(intensity: np.ndarray) -> list[int]:
    peaks: list[int] = []
    if len(intensity) < 3:
        if len(intensity):
            peaks.append(int(np.argmax(intensity)))
        return peaks
    for i in range(1, len(intensity) - 1):
        if intensity[i] >= intensity[i - 1] and intensity[i] > intensity[i + 1]:
            peaks.append(i)
    return peaks


def find_djw_nuclei(
    times_ms: np.ndarray,
    intensity: np.ndarray,
    voiced_flags: list[bool] | None,
    *,
    ignorance_db: float = IGNORANCE_DB,
    min_dip_db: float = MIN_DIP_DB,
) -> list[int]:
    """Intensity-frame indices of DJW syllable nuclei (voiced peaks with dip)."""
    if len(intensity) < 3:
        return []

    med = float(np.median(intensity))
    minint = float(np.min(intensity))
    threshold = med + ignorance_db
    if threshold < minint:
        threshold = minint

    raw_peaks = [i for i in _local_maxima(intensity) if float(intensity[i]) >= threshold]
    if not raw_peaks:
        return []

    # DJW: keep peak[i] if dip from peak[i] down to the min before peak[i+1]
    # exceeds mindip. Last peak gets a symmetric preceding-dip check.
    valid: list[int] = []
    for p in range(len(raw_peaks) - 1):
        i0 = raw_peaks[p]
        i1 = raw_peaks[p + 1]
        lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
        dip = float(np.min(intensity[lo : hi + 1]))
        if abs(float(intensity[i0]) - dip) > min_dip_db:
            valid.append(i0)
    if len(raw_peaks) == 1:
        valid.append(raw_peaks[0])
    else:
        i_last = raw_peaks[-1]
        i_prev = raw_peaks[-2]
        lo, hi = (i_prev, i_last) if i_prev <= i_last else (i_last, i_prev)
        dip = float(np.min(intensity[lo : hi + 1]))
        if abs(float(intensity[i_last]) - dip) > min_dip_db:
            valid.append(i_last)

    valid = sorted(set(valid))
    if voiced_flags is not None:
        valid = [i for i in valid if i < len(voiced_flags) and voiced_flags[i]]
    return valid


def _min_between(times_ms: np.ndarray, intensity: np.ndarray, i0: int, i1: int) -> int:
    lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
    if hi <= lo:
        return lo
    return lo + int(np.argmin(intensity[lo : hi + 1]))


def _trim_active(
    start_ms: float,
    end_ms: float,
    times_ms: np.ndarray,
    intensity: np.ndarray,
    *,
    threshold_db: float,
    pad_ms: float = VOICE_TRIM_PAD_MS,
    min_part_ms: float = RESEG_MIN_PART_MS,
) -> tuple[float, float] | None:
    """Shrink [start,end] to frames above the DJW intensity threshold (+pad)."""
    if len(times_ms) == 0:
        return None
    lo = int(np.searchsorted(times_ms, start_ms, side="left"))
    hi = int(np.searchsorted(times_ms, end_ms, side="right"))
    lo = max(0, min(lo, len(intensity)))
    hi = max(lo, min(hi, len(intensity)))
    if hi <= lo:
        return None
    mask = intensity[lo:hi] >= threshold_db
    if not mask.any():
        if end_ms - start_ms >= min_part_ms:
            return start_ms, end_ms
        return None
    idx = np.where(mask)[0]
    first = lo + int(idx[0])
    last = lo + int(idx[-1])
    t0 = max(start_ms, float(times_ms[first]) - pad_ms)
    t1 = min(end_ms, float(times_ms[last]) + HOP_MS + pad_ms)
    if t1 - t0 < min_part_ms:
        need = min_part_ms - (t1 - t0)
        t0 = max(start_ms, t0 - need / 2)
        t1 = min(end_ms, t0 + min_part_ms)
        if t1 - t0 < min_part_ms * 0.8:
            return None
    return t0, t1


def _force_split_at_dip(
    start_ms: float,
    end_ms: float,
    times_ms: np.ndarray,
    intensity: np.ndarray,
    *,
    target_ms: float,
    min_part_ms: float,
    min_dip_db: float,
    depth: int = 0,
) -> list[tuple[float, float, str | None]]:
    """If a piece is still > target, cut at its deepest interior intensity dip."""
    dur = end_ms - start_ms
    if dur <= target_ms or depth >= 8:
        return [(start_ms, end_ms, None if depth == 0 else "syllable")]

    lo = int(np.searchsorted(times_ms, start_ms, side="left"))
    hi = int(np.searchsorted(times_ms, end_ms, side="right"))
    if hi - lo < 5:
        return [(start_ms, end_ms, None if depth == 0 else "syllable")]

    seg_t = times_ms[lo:hi]
    seg_i = intensity[lo:hi]
    best_j = None
    best_depth = min_dip_db
    for j in range(1, len(seg_i) - 1):
        t = float(seg_t[j])
        if (t - start_ms) < min_part_ms or (end_ms - t) < min_part_ms:
            continue
        if seg_i[j] > seg_i[j - 1] or seg_i[j] > seg_i[j + 1]:
            continue
        left_peak = float(np.max(seg_i[: j + 1]))
        right_peak = float(np.max(seg_i[j:]))
        prom = min(left_peak, right_peak) - float(seg_i[j])
        if prom >= best_depth:
            best_depth = prom
            best_j = j
    if best_j is None:
        return [(start_ms, end_ms, None if depth == 0 else "syllable")]

    mid = float(seg_t[best_j])
    left = _force_split_at_dip(
        start_ms,
        mid,
        times_ms,
        intensity,
        target_ms=target_ms,
        min_part_ms=min_part_ms,
        min_dip_db=min_dip_db,
        depth=depth + 1,
    )
    right = _force_split_at_dip(
        mid,
        end_ms,
        times_ms,
        intensity,
        target_ms=target_ms,
        min_part_ms=min_part_ms,
        min_dip_db=min_dip_db,
        depth=depth + 1,
    )
    if right:
        right[0] = (right[0][0], right[0][1], "syllable")
    if depth == 0 and left:
        left[0] = (left[0][0], left[0][1], None)
    return left + right


def split_span_relative(
    x: np.ndarray,
    sr: int,
    *,
    target_ms: float = RESEG_TARGET_MS,
    min_part_ms: float = RESEG_MIN_PART_MS,
    ignorance_db: float = IGNORANCE_DB,
    min_dip_db: float = MIN_DIP_DB,
    word_split_min_sep_ms: float = WORD_SPLIT_MIN_SEP_MS,
    word_split_min_dip_db: float = WORD_SPLIT_MIN_DIP_DB,
) -> list[tuple[float, float, str | None]]:
    """Split a mono clip (t=0 at clip start) into DJW-nucleus children."""
    x = _mono(x)
    dur_ms = len(x) * 1000.0 / max(sr, 1)
    if dur_ms < RESEG_MIN_PARENT_MS:
        return [(0.0, dur_ms, None)]

    times_ms, intensity = _praat_like_intensity(x, sr)
    med = float(np.median(intensity))
    threshold = max(float(np.min(intensity)), med + ignorance_db)

    raw_peaks = [i for i in _local_maxima(intensity) if float(intensity[i]) >= threshold]
    voiced_at_raw = _voiced_at_peaks(x, sr, times_ms, raw_peaks)
    voiced_flags = [False] * len(intensity)
    for pi, is_v in zip(raw_peaks, voiced_at_raw):
        voiced_flags[pi] = is_v

    nuclei = find_djw_nuclei(
        times_ms,
        intensity,
        voiced_flags,
        ignorance_db=ignorance_db,
        min_dip_db=min_dip_db,
    )

    # Word-like cuts only: DJW nuclei mark syllable beats; we split between
    # nuclei when the gap is long or the valley is deeper than an in-word dip.
    cuts: list[float] = []
    for a, b in zip(nuclei, nuclei[1:]):
        sep_ms = abs(float(times_ms[b]) - float(times_ms[a]))
        mi = _min_between(times_ms, intensity, a, b)
        dip = float(intensity[mi])
        prom = min(float(intensity[a]), float(intensity[b])) - dip
        if sep_ms >= word_split_min_sep_ms or prom >= word_split_min_dip_db:
            cuts.append(float(times_ms[mi]))
    boundaries = [0.0] + cuts + [dur_ms]
    raw_parts = [
        (boundaries[i], boundaries[i + 1], "syllable" if i > 0 else None)
        for i in range(len(boundaries) - 1)
    ]

    parts: list[tuple[float, float, str | None]] = []
    for a, b, reason in raw_parts:
        sub = _force_split_at_dip(
            a,
            b,
            times_ms,
            intensity,
            target_ms=target_ms,
            min_part_ms=min_part_ms,
            min_dip_db=FORCE_SPLIT_MIN_DIP_DB,
        )
        if reason and sub and sub[0][2] is None:
            sub[0] = (sub[0][0], sub[0][1], reason)
        parts.extend(sub)

    trimmed: list[tuple[float, float, str | None]] = []
    for a, b, reason in parts:
        tr = _trim_active(
            a, b, times_ms, intensity, threshold_db=threshold, min_part_ms=min_part_ms
        )
        if tr is None:
            continue
        trimmed.append((tr[0], tr[1], reason))
    return trimmed or [(0.0, dur_ms, None)]


def resegment_pieces(
    pieces: Iterable[dict],
    audio: np.ndarray,
    sr: int,
    *,
    enabled: bool = True,
    target_ms: float = RESEG_TARGET_MS,
    min_part_ms: float = RESEG_MIN_PART_MS,
    min_gap_ms: float | None = None,  # API compat; unused (DJW uses dip dB)
    min_prominence_db: float | None = None,  # API compat → min_dip override
    ignorance_db: float = IGNORANCE_DB,
    min_dip_db: float = MIN_DIP_DB,
    word_split_min_sep_ms: float = WORD_SPLIT_MIN_SEP_MS,
    word_split_min_dip_db: float = WORD_SPLIT_MIN_DIP_DB,
) -> tuple[list[dict], dict]:
    """Expand parent pieces into DJW-nucleus children."""
    del min_gap_ms  # unused
    if min_prominence_db is not None:
        min_dip_db = float(min_prominence_db)

    stats = {
        "resegParents": 0,
        "resegSplits": 0,
        "resegTrimmedOnly": 0,
        "resegSkipped": 0,
        "resegChildren": 0,
        "resegMethod": "dejong_wempe",
        "wordSplitMinSepMs": word_split_min_sep_ms,
        "wordSplitMinDipDb": word_split_min_dip_db,
        "resegTargetMs": target_ms,
        "resegMinPartMs": min_part_ms,
    }
    audio = _mono(audio)
    if not enabled:
        out = list(pieces)
        stats["resegChildren"] = len(out)
        stats["resegSkipped"] = len(out)
        return out, stats

    out: list[dict] = []
    for piece in pieces:
        start = float(piece["start"])
        end = float(piece["end"])
        dur = end - start
        s = max(0, int(start / 1000.0 * sr))
        e = min(len(audio), int(end / 1000.0 * sr))
        if e <= s:
            stats["resegSkipped"] += 1
            continue

        clip = audio[s:e]
        parent_id = str(uuid.uuid4())
        stats["resegParents"] += 1
        parent_split = (piece.get("meta") or {}).get("splitBy")

        if dur < RESEG_MIN_PARENT_MS:
            times_ms, intensity = _praat_like_intensity(clip, sr)
            med = float(np.median(intensity))
            threshold = max(float(np.min(intensity)), med + ignorance_db)
            tr = _trim_active(
                0.0, dur, times_ms, intensity, threshold_db=threshold, min_part_ms=min_part_ms
            )
            if tr is None:
                stats["resegSkipped"] += 1
                continue
            meta = dict(piece.get("meta") or {})
            if meta.get("flags"):
                meta["flags"] = list(meta["flags"])
            meta["parentSpanId"] = parent_id
            meta["resegMethod"] = "dejong_wempe"
            if parent_split:
                meta["splitBy"] = parent_split
            if abs(tr[0]) > 1 or abs(tr[1] - dur) > 1:
                stats["resegTrimmedOnly"] += 1
            out.append(
                {
                    "start": start + tr[0],
                    "end": start + tr[1],
                    "dur": tr[1] - tr[0],
                    "score": piece.get("score", 0.0),
                    "meta": meta,
                }
            )
            continue

        rel_parts = split_span_relative(
            clip,
            sr,
            target_ms=target_ms,
            min_part_ms=min_part_ms,
            ignorance_db=ignorance_db,
            min_dip_db=min_dip_db,
            word_split_min_sep_ms=word_split_min_sep_ms,
            word_split_min_dip_db=word_split_min_dip_db,
        )
        if len(rel_parts) > 1:
            stats["resegSplits"] += len(rel_parts) - 1
        elif len(rel_parts) == 1 and (
            abs(rel_parts[0][0]) > 1 or abs(rel_parts[0][1] - dur) > 1
        ):
            stats["resegTrimmedOnly"] += 1

        for i, (a, b, reason) in enumerate(rel_parts):
            meta = dict(piece.get("meta") or {})
            if meta.get("flags"):
                meta["flags"] = list(meta["flags"])
            meta["parentSpanId"] = parent_id
            meta["resegMethod"] = "dejong_wempe"
            if reason:
                meta["splitBy"] = reason
            elif i > 0:
                meta["splitBy"] = "syllable"
            elif parent_split:
                meta["splitBy"] = parent_split
            out.append(
                {
                    "start": start + a,
                    "end": start + b,
                    "dur": b - a,
                    "score": piece.get("score", 0.0),
                    "meta": meta,
                }
            )

    stats["resegChildren"] = len(out)
    return out, stats


# ---------------------------------------------------------------------------
# Short-gated merge-back (word-like candidates for Review)
# ---------------------------------------------------------------------------


def _piece_intensity(
    audio: np.ndarray, sr: int, start_ms: float, end_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    s = max(0, int(start_ms / 1000.0 * sr))
    e = min(len(audio), int(end_ms / 1000.0 * sr))
    if e <= s:
        return np.asarray([0.0]), np.asarray([-80.0])
    times, intensity = _praat_like_intensity(audio[s:e], sr)
    return times + start_ms, intensity


def _region_median(
    times: np.ndarray, intensity: np.ndarray, lo: float, hi: float
) -> float:
    if hi <= lo or times.size == 0:
        return -80.0
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        mid = 0.5 * (lo + hi)
        i = int(np.argmin(np.abs(times - mid)))
        return float(intensity[i])
    return float(np.median(intensity[mask]))


def _cut_features(
    audio: np.ndarray,
    sr: int,
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
    params: MergeBackParams,
) -> dict:
    a0, a1 = float(left_start), float(left_end)
    b0, b1 = float(right_start), float(right_end)
    gap = b0 - a1
    merged_dur = b1 - a0

    times, intensity = _piece_intensity(audio, sr, a0, b1)
    left_med = _region_median(times, intensity, a0, a1)
    right_med = _region_median(times, intensity, b0, b1)
    cut = 0.5 * (a1 + b0)
    win = params.cut_window_ms
    gap_lo = min(a1, b0)
    gap_hi = max(a1, b0)
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
    *,
    parent_key: str = "parentSpanId",
    speaker_key: str = "speakerCluster",
) -> tuple[bool, str]:
    """Decide keep-cut vs merge for two adjacent sibling spans."""
    pid_l = left.get(parent_key)
    pid_r = right.get(parent_key)
    if pid_l and pid_r:
        if pid_l != pid_r:
            return False, "diff_parent"
    else:
        return False, "missing_parent"

    sc_l = left.get(speaker_key)
    sc_r = right.get(speaker_key)
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


def _merge_piece_pair(left: dict, right: dict) -> dict:
    """Merge two pipeline pieces (start/end/meta)."""
    meta_l = dict(left.get("meta") or {})
    meta_r = dict(right.get("meta") or {})
    meta = dict(meta_l)
    flags = list(meta_l.get("flags") or [])
    for f in meta_r.get("flags") or []:
        if f not in flags:
            flags.append(f)
    if "merge_back" not in flags:
        flags.append("merge_back")
    meta["flags"] = flags
    meta["splitBy"] = "merge_back"
    meta["resegMethod"] = meta.get("resegMethod") or meta_r.get("resegMethod") or "dejong_wempe"
    # Preserve parent / speaker from the left sibling.
    if meta_l.get("parentSpanId"):
        meta["parentSpanId"] = meta_l["parentSpanId"]
    merged_from = list(meta_l.get("mergedFrom") or [])
    if not merged_from:
        # First merge: invent stable ids for the original children if absent.
        lid = meta_l.get("pieceId") or str(uuid.uuid4())
        rid = meta_r.get("pieceId") or str(uuid.uuid4())
        merged_from = [lid, rid]
    else:
        rid = meta_r.get("pieceId") or str(uuid.uuid4())
        if meta_r.get("mergedFrom"):
            for mid in meta_r["mergedFrom"]:
                if mid not in merged_from:
                    merged_from.append(mid)
        elif rid not in merged_from:
            merged_from.append(rid)
    meta["mergedFrom"] = merged_from

    scores = [left.get("score"), right.get("score")]
    scores = [s for s in scores if s is not None]
    speech_scores = [meta_l.get("speechScore"), meta_r.get("speechScore")]
    speech_scores = [s for s in speech_scores if s is not None]
    if speech_scores:
        meta["speechScore"] = max(speech_scores)

    start = float(left["start"])
    end = float(right["end"])
    out = {
        "start": start,
        "end": end,
        "dur": end - start,
        "score": max(scores) if scores else left.get("score", 0.0),
        "meta": meta,
    }
    # Carry left annotation template when operating on annotation-shaped inputs.
    if "_ann" in left:
        out["_ann"] = left["_ann"]
    return out


def merge_back_pieces(
    pieces: list[dict],
    audio: np.ndarray,
    sr: int,
    params: MergeBackParams | None = None,
) -> tuple[list[dict], dict]:
    """Greedy left-to-right merge of sibling DJW children (piece format).

    Pieces must carry ``meta.parentSpanId`` (set by ``resegment_pieces``).
    Same-speaker and energy guards apply. Returns word-like candidates.
    """
    params = params or MergeBackParams()
    audio = _mono(audio)
    by_parent: dict[str, list[dict]] = {}
    orphans: list[dict] = []
    for p in pieces:
        meta = p.get("meta") or {}
        pid = meta.get("parentSpanId")
        if not pid:
            orphans.append(dict(p))
            continue
        by_parent.setdefault(str(pid), []).append(dict(p))

    stats = {
        "mergeBackIn": len(pieces),
        "mergeBackParents": len(by_parent),
        "mergeBackMerges": 0,
        "mergeBackConsidered": 0,
        "mergeBackRejectReasons": {},
        "mergeBackMergeReasons": {},
        "mergeBackShortPieceMs": params.short_piece_ms,
        "mergeBackMaxGapMs": params.max_gap_ms,
        "mergeBackRequireClearlyShort": params.require_clearly_short,
        "mergeBackEnabled": True,
    }

    out: list[dict] = list(orphans)
    for _pid, sibs in by_parent.items():
        sibs = sorted(sibs, key=lambda x: (float(x["start"]), float(x["end"])))
        if len(sibs) == 1:
            out.append(sibs[0])
            continue
        acc = sibs[0]
        for nxt in sibs[1:]:
            meta_a = acc.get("meta") or {}
            meta_b = nxt.get("meta") or {}
            feats = _cut_features(
                audio,
                sr,
                float(acc["start"]),
                float(acc["end"]),
                float(nxt["start"]),
                float(nxt["end"]),
                params,
            )
            stats["mergeBackConsidered"] += 1
            left_view = {
                "parentSpanId": meta_a.get("parentSpanId"),
                "speakerCluster": meta_a.get("speakerCluster"),
            }
            right_view = {
                "parentSpanId": meta_b.get("parentSpanId"),
                "speakerCluster": meta_b.get("speakerCluster"),
            }
            ok, reason = should_merge(left_view, right_view, feats, params)
            if ok:
                acc = _merge_piece_pair(acc, nxt)
                stats["mergeBackMerges"] += 1
                stats["mergeBackMergeReasons"][reason] = (
                    stats["mergeBackMergeReasons"].get(reason, 0) + 1
                )
            else:
                stats["mergeBackRejectReasons"][reason] = (
                    stats["mergeBackRejectReasons"].get(reason, 0) + 1
                )
                out.append(acc)
                acc = nxt
        out.append(acc)

    out.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    stats["mergeBackOut"] = len(out)
    return out, stats


def apply_merge_back(
    cands: list[dict],
    audio: np.ndarray,
    sr: int,
    params: MergeBackParams | None = None,
) -> tuple[list[dict], dict]:
    """Merge-back on annotation-shaped dicts (``startMs``/``endMs``).

    Used by offline analysis; production path uses ``merge_back_pieces``.
    """
    params = params or MergeBackParams()
    pieces: list[dict] = []
    for c in cands:
        meta = {
            "parentSpanId": c.get("parentSpanId"),
            "speakerCluster": c.get("speakerCluster"),
            "flags": list(c.get("flags") or []),
            "speechScore": c.get("speechScore"),
            "pieceId": c.get("uuid"),
            "splitBy": c.get("splitBy"),
            "resegMethod": c.get("resegMethod"),
        }
        if c.get("mergedFrom"):
            meta["mergedFrom"] = list(c["mergedFrom"])
        pieces.append(
            {
                "start": float(c["startMs"]),
                "end": float(c["endMs"]),
                "dur": float(c["endMs"]) - float(c["startMs"]),
                "score": c.get("score", c.get("speechScore", 0.0)),
                "meta": meta,
                "_ann": c,
            }
        )

    merged_pieces, stats = merge_back_pieces(pieces, audio, sr, params)

    # Remap stats keys to the analysis-script names for compatibility.
    compat = {
        "nIn": stats["mergeBackIn"],
        "nParents": stats["mergeBackParents"],
        "nMerges": stats["mergeBackMerges"],
        "nConsidered": stats["mergeBackConsidered"],
        "rejectReasons": stats["mergeBackRejectReasons"],
        "mergeReasons": stats["mergeBackMergeReasons"],
        "nOut": stats["mergeBackOut"],
    }

    out: list[dict] = []
    for p in merged_pieces:
        base = p.get("_ann")
        meta = p.get("meta") or {}
        if base is not None and not meta.get("mergedFrom"):
            # Unchanged orphan / singleton — keep original annotation.
            out.append(dict(base))
            continue
        ann = dict(base) if base is not None else {}
        ann["uuid"] = str(uuid.uuid4())
        ann["startMs"] = int(round(p["start"]))
        ann["endMs"] = int(round(p["end"]))
        ann["tMs"] = ann["startMs"]
        if p.get("score") is not None:
            ann["score"] = p["score"]
        if meta.get("speechScore") is not None:
            ann["speechScore"] = meta["speechScore"]
        if meta.get("parentSpanId"):
            ann["parentSpanId"] = meta["parentSpanId"]
        if meta.get("flags"):
            ann["flags"] = list(meta["flags"])
        if meta.get("splitBy"):
            ann["splitBy"] = meta["splitBy"]
        if meta.get("mergedFrom"):
            ann["mergedFrom"] = list(meta["mergedFrom"])
        out.append(ann)

    compat["nOut"] = len(out)
    return out, compat
