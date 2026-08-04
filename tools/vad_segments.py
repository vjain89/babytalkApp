"""ML candidate pipeline for BabyTalk session kits (vad_v0 + diar_v1 / vtc).

Stages turn a raw session recording into provisional annotations for Review.

**Default path (VAD-first):**

  1. **VAD (this module)** — energy detection of louder-than-the-room regions,
     then a *speech-likeness* gate (``speechlike.py``) that asks whether each
     region actually sounds like a vocal tract — voicing, voice-band energy,
     low-frequency rumble — rather than a tap, a door or a chair scrape.
     Silence is dropped, obvious non-speech is dropped before diarization, and
     each final candidate is re-judged individually.
  2. **Speaker diarization (``diarize.py``)** — speaker embeddings over sliding
     windows inside those regions, clustered across the whole recording, then
     each region is cut wherever the speaker changes. Tags each piece with a
     ``SPEAKER_xx`` cluster id. Pass ``--diarization vtc`` to cut on Voice Type
     Classifier roles (KCHI/OCH/FEM/MAL) instead of ECAPA clusters.
  3. **Refine (this module + ``resegment.py``)** — same-speaker spans still
     over 4s are pause-split; then each span is cut with de Jong & Wempe
     nuclei and **short-gated merge-back** into word-like (tag-sized) children
     — not raw syllable shards. An absolute duration cap is a last resort.
     Candidates overlapping ``tags.json`` are dropped. Results write to
     ``annotations.json`` as provisional candidates.

**VTC-first path (``--segmentation vtc-first``):**

  Skip energy VAD + ECAPA. Run the LAAC-LSCP Voice Type Classifier on the whole
  file; its role turns become the parent spans (with ``speaker`` prefilled as
  Baby/Parent/Other). Pause-split, DJW + merge-back, and the speechlike gate
  still run. See ``vtc.py``.

Stage 2 degrades gracefully: if no diarization backend is usable (or
``--no-diarization`` is passed) the pipeline falls back to VAD + pause /
word-like resegmentation only, and says so in the returned stats.

Requires: numpy, soundfile (stage 1) · see diarize.py / vtc.py for stage-2 deps
  pip install numpy soundfile

Usage:
  python3 tools/vad_segments.py /path/to/kit_or_library
  python3 tools/vad_segments.py /path/to/kit --diarization ecapa --num-speakers 2
  python3 tools/vad_segments.py /path/to/kit --segmentation vtc-first --dry-run
  python3 tools/vad_segments.py /path/to/kit --no-diarization --dry-run
  python3 tools/vad_segments.py /path/to/kit --no-resegment
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install deps: pip install numpy soundfile\n" + str(e)
    ) from e

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import resegment as reseg_mod
import speechlike

# Defaults tuned for baby/caregiver utterances reviewed in the Mac UI.
FRAME_MS = 30
HOP_MS = 10
# Speech if frame is this many dB above a robust noise floor.
SPEECH_DELTA_DB = 6.0
# Pad each run so soft onsets/offsets aren't clipped.
PAD_MS = 80
# --- Onset/offset hysteresis -----------------------------------------------
# A run only *starts* once a frame clears SPEECH_DELTA_DB (that's what keeps
# the gate from false-triggering on room noise), but a real attack or release
# ramps up/down gradually — several frames right next to the run are often
# already well above the noise floor without (yet) being "confidently loud".
# A single fixed threshold treats all of that ramp as silence and only PAD_MS
# stands between the detected edge and the true one, which is exactly the
# "starts after the word starts / ends before it ends" clipping this guards
# against. So before padding, each run's edges are walked back/forward
# through neighboring frames that clear a *lower* threshold — still a level
# gate, just a second, more lenient one, not a derivative/onset detector —
# so a slow onset can "reach back" to where the sound actually left the
# floor. Bounded in both distance and by the previous/next run's own edge so
# this can never stretch through a real gap into a neighboring utterance.
ONSET_LOW_DELTA_DB = 3.0
ONSET_MAX_EXTEND_MS = 120
# Merge segments separated by ≤ this gap (same utterance).
# Keep short so parent/baby turn-taking doesn't glue into one candidate.
MERGE_GAP_MS = 200
# Drop segments shorter than this (likely clicks/noise).
MIN_DUR_MS = 300
# After diarization, a single-speaker span longer than this still gets a
# pause-based re-split (one speaker, several sentences).
SPLIT_TARGET_MS = 4_000
# Never split into pieces shorter than this.
SPLIT_MIN_PART_MS = 600
# A candidate energy valley must dip at least this many dB below the lower
# of its two flanking peaks to count as a plausible utterance boundary.
SPLIT_MIN_PROMINENCE_DB = 5.0
# Absolute last resort if nothing above could split a long stretch (e.g. one
# continuous monologue with no clean pause) — keep first N ms only.
HARD_MAX_DUR_MS = 15_000
# --- Non-speech rejection --------------------------------------------------
# Two independent checks, because they catch different things:
#
#   * The *absolute* speech-likeness gate in speechlike.py is the primary one.
#     It judges each segment on its own acoustics (voicing, voice-band energy,
#     sub-250 Hz rumble) and is calibrated against spans this project's
#     reviewer actually confirmed vs dismissed.
#   * The *relative* impulsive check below stays as a second opinion for very
#     short bursts, where a click can still land inside the voice band. It
#     z-scores flatness/ZCR/"spikiness" against the other candidates in the
#     same recording, so it only ever flags a recording's own outliers — that
#     narrowness is exactly why it can't be the primary gate.
#
# Only consider rejecting segments this short as "impulsive" (taps/door
# closes are brief; genuine multi-syllable speech usually runs longer).
NONSPEECH_MAX_DUR_MS = 700
# Fraction of frames within SUSTAIN_WINDOW_DB of the segment's peak. Speech
# syllables have a "hump" (several frames near the peak); a sharp impulse
# decays away almost immediately, so this ratio stays low.
SUSTAIN_WINDOW_DB = 6.0
# How far above this file's own median (in z-score units, across flatness,
# ZCR, and inverse-sustain) a short segment must be before we treat it as
# noise-like rather than speech-like.
NONSPEECH_COMPOSITE_Z = 1.0
# Need at least this many short candidates before z-scoring is meaningful;
# below this, skip the relative check (too little data to compare against).
NONSPEECH_MIN_SAMPLES = 5
SOURCE = "vad_v0"
# --- Tag-overlap suppression ----------------------------------------------
# A re-run of VAD should never re-propose a span a human (or a confirmed-ML
# candidate) already tagged. Interval overlap is checked with this small
# margin on both sides so a candidate ending 30ms before a tag starts isn't
# treated as touching it, while still catching near-duplicate edges.
TAG_OVERLAP_MARGIN_MS = 75.0
# Some tags are effectively instantaneous (endMs is null — a point mark, not
# a reviewed span). We can't know the "true" duration of that sound, so we
# treat it as covering a short window starting at its timestamp; anything
# that overlaps that window is suppressed rather than trying to guess where
# the sound actually ended.
TAG_POINT_SPAN_MS = 500.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def frame_rms_db(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Cheap whole-track RMS-per-frame (dB), via a cumulative-sum trick so it
    stays O(n) instead of materializing an (n_frames x frame_len) matrix.
    """
    frame = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    if n < frame:
        rms = float(np.sqrt(np.mean(audio * audio)) + 1e-12) if n else 1e-12
        return np.asarray([0.0]), np.asarray([20.0 * math.log10(rms)])
    n_frames = (n - frame) // hop + 1
    starts = hop * np.arange(n_frames)
    cums = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
    sums = cums[starts + frame] - cums[starts]
    rms = np.sqrt(sums / frame) + 1e-12
    dbs = 20.0 * np.log10(rms)
    times = starts * 1000.0 / sr
    return times, dbs


def frame_features(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Vectorized per-frame features for a (typically short) audio slice:
    RMS (dB), zero-crossing rate, spectral flatness, spectral centroid.

    Only meant to be called on bounded-size slices (single candidate
    segments), not the whole recording — building the (frames x frame_len)
    matrix is O(n_frames * frame_len) in memory.
    """
    frame = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float64)
    n = len(audio)
    if n < frame:
        padded = np.zeros(max(frame, 1), dtype=np.float64)
        padded[:n] = audio
        frames = padded[None, :]
        starts = np.asarray([0.0])
    else:
        n_frames = (n - frame) // hop + 1
        idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = audio[idx]
        starts = (hop * np.arange(n_frames)).astype(np.float64)

    times = starts * 1000.0 / sr
    rms = np.sqrt(np.mean(frames * frames, axis=1)) + 1e-12
    dbs = 20.0 * np.log10(rms)

    signs = frames >= 0
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    window = np.hanning(frame) if frame > 1 else np.ones(frame)
    spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1)) + 1e-9
    flatness = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)
    centroid = (spec * freqs[None, :]).sum(axis=1) / spec.sum(axis=1)

    return {
        "times": times,
        "rms_db": dbs,
        "zcr": zcr,
        "flatness": flatness,
        "centroid": centroid,
    }


def noise_floor_db(dbs: np.ndarray) -> float:
    """Robust floor: lower quartile of frames (silence-heavy sessions)."""
    if len(dbs) == 0:
        return -80.0
    return float(np.percentile(dbs, 25))


def _idx_range(times: np.ndarray, start_ms: float, end_ms: float) -> tuple[int, int]:
    lo = int(np.searchsorted(times, start_ms, side="left"))
    hi = int(np.searchsorted(times, end_ms, side="right"))
    return lo, hi


def _best_energy_valley(
    times: np.ndarray,
    dbs: np.ndarray,
    lo: int,
    hi: int,
    *,
    min_part_ms: float,
    min_prominence_db: float,
) -> float | None:
    """Deepest interior local-minimum dB point in [lo, hi) that leaves both
    sides >= min_part_ms and dips >= min_prominence_db below its flanks.
    Returns a ms position, or None if no plausible valley exists.
    """
    if hi - lo < 5:
        return None
    seg = dbs[lo:hi]
    seg_t = times[lo:hi]
    k = 3
    if len(seg) >= k:
        smooth = np.convolve(seg, np.ones(k) / k, mode="same")
    else:
        smooth = seg

    best_i = None
    best_prom = min_prominence_db
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] or smooth[i] > smooth[i + 1]:
            continue
        t = float(seg_t[i])
        if (t - seg_t[0]) < min_part_ms or (seg_t[-1] - t) < min_part_ms:
            continue
        left_peak = float(np.max(smooth[: i + 1]))
        right_peak = float(np.max(smooth[i:]))
        prominence = min(left_peak, right_peak) - float(smooth[i])
        if prominence >= best_prom:
            best_prom = prominence
            best_i = i
    if best_i is None:
        return None
    return float(seg_t[best_i])


def _split_on_pauses(
    start_ms: float,
    end_ms: float,
    times: np.ndarray,
    dbs: np.ndarray,
    *,
    split_target_ms: float,
    min_part_ms: float,
    min_prominence_db: float,
    depth: int = 0,
) -> list[tuple[float, float, str | None]]:
    """Recursively split a long span at its deepest internal pause.

    This runs *after* diarization, so it is only ever separating consecutive
    utterances by the same speaker — a pause is a decent utterance boundary,
    and unlike the old spectral-shift heuristic it makes no claim about who is
    talking. Returns [(start_ms, end_ms, split_reason)] where split_reason
    describes why this piece was cut from its predecessor.
    """
    dur = end_ms - start_ms
    if dur <= split_target_ms or depth >= 6:
        return [(start_ms, end_ms, None)]

    lo, hi = _idx_range(times, start_ms, end_ms)
    split_ms = _best_energy_valley(
        times, dbs, lo, hi, min_part_ms=min_part_ms, min_prominence_db=min_prominence_db
    )
    if split_ms is None:
        return [(start_ms, end_ms, None)]

    kwargs = dict(
        split_target_ms=split_target_ms,
        min_part_ms=min_part_ms,
        min_prominence_db=min_prominence_db,
        depth=depth + 1,
    )
    left = _split_on_pauses(start_ms, split_ms, times, dbs, **kwargs)
    right = _split_on_pauses(split_ms, end_ms, times, dbs, **kwargs)
    right[0] = (right[0][0], right[0][1], "pause")
    return left + right


def _segment_shape_features(audio: np.ndarray, sr: int, start_ms: float, end_ms: float) -> dict:
    s = max(0, int(start_ms / 1000.0 * sr))
    e = min(len(audio), int(end_ms / 1000.0 * sr))
    if e <= s:
        return {"flatness": 0.0, "zcr": 0.0, "sustain_ratio": 1.0}
    feats = frame_features(audio[s:e], sr)
    dbs = feats["rms_db"]
    if len(dbs) == 0:
        return {"flatness": 0.0, "zcr": 0.0, "sustain_ratio": 1.0}
    peak = float(np.max(dbs))
    sustain_ratio = float(np.mean(dbs >= peak - SUSTAIN_WINDOW_DB))
    return {
        "flatness": float(np.mean(feats["flatness"])),
        "zcr": float(np.mean(feats["zcr"])),
        "sustain_ratio": sustain_ratio,
    }


def _speech_score_for_span(
    audio: np.ndarray, sr: int, start_ms: float, end_ms: float
) -> tuple[float, dict]:
    """Absolute speech-likeness of one span (see ``speechlike.py``)."""
    s = max(0, int(start_ms / 1000.0 * sr))
    e = min(len(audio), int(end_ms / 1000.0 * sr))
    if e <= s:
        return 0.0, {}
    return speechlike.speech_likeness(audio[s:e], sr)


def _non_speech_scores(shapes: list[dict], durations: list[float]) -> list[float]:
    """Composite "sounds like an impulsive non-speech event" z-score per
    segment, relative to the *other candidates in this same recording*
    (device/room acoustics shift absolute flatness/ZCR a lot, so a fixed
    global cutoff is unreliable — see module docstring).

    Combines: spectral flatness (higher = more broadband/noise-like),
    zero-crossing rate (higher = more click-like), and inverse energy
    "sustain" (lower sustain = a sharp spike rather than a speech hump).
    Only meaningful for short segments — longer runs are set to -inf so
    they're never flagged as impulsive by this check.
    """
    n = len(shapes)
    if n < NONSPEECH_MIN_SAMPLES:
        return [float("-inf")] * n

    def _z(vals: np.ndarray) -> np.ndarray:
        std = float(vals.std())
        if std < 1e-9:
            return np.zeros_like(vals)
        return (vals - float(vals.mean())) / std

    flat = np.asarray([s["flatness"] for s in shapes])
    zcr = np.asarray([s["zcr"] for s in shapes])
    spikiness = 1.0 - np.asarray([s["sustain_ratio"] for s in shapes])
    composite = (_z(flat) + _z(zcr) + _z(spikiness)) / 3.0

    out = []
    for i, dur in enumerate(durations):
        out.append(float(composite[i]) if dur <= NONSPEECH_MAX_DUR_MS else float("-inf"))
    return out


# --------------------------------------------------------------------------
# Stage 1 — VAD: speech vs. silence / impulsive noise
# --------------------------------------------------------------------------


def detect_speech_regions(
    times: np.ndarray,
    dbs: np.ndarray,
    *,
    audio: np.ndarray | None = None,
    sr: int | None = None,
    speech_delta_db: float = SPEECH_DELTA_DB,
    pad_ms: float = PAD_MS,
    onset_low_delta_db: float = ONSET_LOW_DELTA_DB,
    onset_max_extend_ms: float = ONSET_MAX_EXTEND_MS,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_dur_ms: float = MIN_DUR_MS,
    reject_non_speech: bool = True,
) -> tuple[list[dict], dict]:
    """Stage 1 only: where is there speech-like activity?

    Returns ``[{start, end, score, meta}]`` regions (no speaker knowledge, no
    length limit) plus a stats dict. Whether a 6s region holds one utterance
    or four turns is stage 2's problem.

    Passing ``audio``/``sr`` enables the spectral-shape non-speech rejection;
    without them this is plain energy-threshold VAD.
    """
    stats = {"rawRuns": 0, "merged": 0, "nonSpeechRejected": 0, "regionsScreened": 0}
    if len(dbs) == 0:
        return [], stats

    floor = noise_floor_db(dbs)
    thresh = floor + speech_delta_db
    active = dbs >= thresh
    # Lenient second threshold used only to extend an already-found run's
    # edges into its own attack/decay ramp (see ONSET_LOW_DELTA_DB above).
    low_thresh = floor + max(0.0, speech_delta_db - onset_low_delta_db)
    elevated = dbs >= low_thresh
    extend_frames = max(0, int(round(onset_max_extend_ms / HOP_MS)))

    raw: list[tuple[float, float, float]] = []
    i = 0
    n = len(active)
    prev_run_end = -1  # index just past the previous run's extended tail
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        peak = float(dbs[i])
        while j < n and active[j]:
            peak = max(peak, float(dbs[j]))
            j += 1

        onset_i = i
        onset_limit = max(prev_run_end, i - extend_frames)
        while onset_i > onset_limit and elevated[onset_i - 1]:
            onset_i -= 1

        offset_j = j
        offset_limit = min(n, j + extend_frames)
        while offset_j < offset_limit and elevated[offset_j]:
            offset_j += 1

        start_ms = float(times[onset_i]) - pad_ms
        end_ms = float(times[min(offset_j - 1, n - 1)]) + FRAME_MS + pad_ms
        score = float(min(1.0, max(0.0, (peak - thresh) / 18.0)))
        raw.append((start_ms, end_ms, score))
        prev_run_end = offset_j
        i = max(j, i + 1)
    stats["rawRuns"] = len(raw)

    if not raw:
        return [], stats

    merged: list[tuple[float, float, float]] = [raw[0]]
    for start_ms, end_ms, score in raw[1:]:
        ps, pe, pscore = merged[-1]
        if start_ms - pe <= merge_gap_ms:
            merged[-1] = (ps, end_ms, max(pscore, score))
        else:
            merged.append((start_ms, end_ms, score))
    stats["merged"] = len(merged)

    regions: list[dict] = []
    for start_ms, end_ms, score in merged:
        start_ms = max(0.0, start_ms)
        end_ms = max(start_ms, end_ms)
        dur = end_ms - start_ms
        if dur < min_dur_ms:
            continue
        regions.append({"start": start_ms, "end": end_ms, "dur": dur, "score": score, "meta": {}})

    # Absolute speech-likeness screen, before anything expensive. Lenient on
    # purpose: a region can hold speech *and* noise, and every surviving piece
    # is re-judged individually after diarization. The point here is to keep
    # obvious junk out of the speaker clustering.
    if audio is not None and sr and reject_non_speech:
        kept_regions: list[dict] = []
        for r in regions:
            score = _speech_score_for_span(audio, sr, r["start"], r["end"])[0]
            if score < speechlike.SPEECH_SCORE_REGION_REJECT:
                stats["regionsScreened"] += 1
                continue
            kept_regions.append(r)
        regions = kept_regions

    # Non-speech scoring is relative to the other regions found in this same
    # recording (see _non_speech_scores for why it's relative, not absolute).
    if audio is not None and sr:
        shapes = [_segment_shape_features(audio, sr, r["start"], r["end"]) for r in regions]
        ns_scores = _non_speech_scores(shapes, [r["dur"] for r in regions])
        kept: list[dict] = []
        for r, ns in zip(regions, ns_scores):
            if ns != float("-inf") and ns >= NONSPEECH_COMPOSITE_Z:
                if reject_non_speech:
                    stats["nonSpeechRejected"] += 1
                    continue
                r["score"] = min(r["score"], 0.35)
                r["meta"]["flags"] = r["meta"].get("flags", []) + ["possible_non_speech"]
            kept.append(r)
        regions = kept

    return regions, stats


# --------------------------------------------------------------------------
# Stage 2 — speaker diarization: cut regions where the speaker changes
# --------------------------------------------------------------------------


def split_by_speaker(
    regions: list[dict],
    audio: np.ndarray,
    sr: int,
    *,
    backend: str = "auto",
    num_speakers: int | None = None,
    distance: float | None = None,
) -> tuple[list[dict], dict]:
    """Run diarization over the stage-1 regions and cut them into
    speaker-homogeneous pieces.

    Returns ``(pieces, info)``. On any failure (no backend, model download
    problem, unexpected model error) the regions are returned unchanged with
    ``info["ok"] is False`` and a human-readable ``info["error"]`` — the
    caller then continues with VAD-only candidates.
    """
    info: dict = {"ok": False, "backend": "none", "speakerSplits": 0, "numSpeakers": 0}
    if not regions:
        info["ok"] = True
        return regions, info

    try:
        from diarize import diarize_regions
    except ImportError as e:
        info["error"] = f"diarize.py unavailable: {e}"
        return regions, info

    result = diarize_regions(
        audio,
        sr,
        [(r["start"], r["end"]) for r in regions],
        backend=backend,
        num_speakers=num_speakers,
        distance=distance,
    )
    info["backend"] = result.backend
    if not result.ok:
        info["error"] = result.error or "diarization failed"
        return regions, info

    # Map each turn back onto the region it came from so we keep the region's
    # VAD score and any stage-1 flags.
    pieces: list[dict] = []
    turns_by_region: dict[int, list] = {}
    bounds = [(r["start"], r["end"]) for r in regions]
    for turn in result.turns:
        mid = 0.5 * (turn.start_ms + turn.end_ms)
        owner = None
        for ri, (rs, re_) in enumerate(bounds):
            if rs - 1e-6 <= mid <= re_ + 1e-6:
                owner = ri
                break
        if owner is None:
            continue
        turns_by_region.setdefault(owner, []).append(turn)

    for ri, region in enumerate(regions):
        turns = sorted(turns_by_region.get(ri, []), key=lambda t: t.start_ms)
        if not turns:
            pieces.append(region)
            continue
        if len(turns) > 1:
            info["speakerSplits"] += len(turns) - 1
        for ti, turn in enumerate(turns):
            meta = dict(region["meta"])
            if meta.get("flags"):
                meta["flags"] = list(meta["flags"])
            if turn.speaker:
                meta["speakerCluster"] = turn.speaker
                meta["speakerConfidence"] = round(float(turn.confidence), 3)
            if ti > 0:
                meta["splitBy"] = "speaker_change"
            pieces.append(
                {
                    "start": turn.start_ms,
                    "end": turn.end_ms,
                    "dur": turn.end_ms - turn.start_ms,
                    "score": region["score"],
                    "meta": meta,
                }
            )

    info["ok"] = True
    info["numSpeakers"] = result.num_speakers
    info["stats"] = result.stats
    return pieces, info


# --------------------------------------------------------------------------
# Stage 3 — refine long spans and emit candidates
# --------------------------------------------------------------------------


def refine_long_spans(
    pieces: list[dict],
    times: np.ndarray,
    dbs: np.ndarray,
    *,
    split_target_ms: float = SPLIT_TARGET_MS,
    min_dur_ms: float = MIN_DUR_MS,
    max_dur_ms: float = HARD_MAX_DUR_MS,
    split_min_part_ms: float = SPLIT_MIN_PART_MS,
    split_min_prominence_db: float = SPLIT_MIN_PROMINENCE_DB,
) -> tuple[list[dict], dict]:
    """Pause-split anything still longer than ``split_target_ms``, then apply
    the absolute duration cap. Runs per speaker turn, so it never merges
    across a speaker boundary.
    """
    stats = {"pauseSplit": 0, "hardCapped": 0}
    out: list[dict] = []
    for piece in pieces:
        parts = _split_on_pauses(
            piece["start"],
            piece["end"],
            times,
            dbs,
            split_target_ms=split_target_ms,
            min_part_ms=split_min_part_ms,
            min_prominence_db=split_min_prominence_db,
        )
        if len(parts) > 1:
            stats["pauseSplit"] += len(parts) - 1
        for part_start, part_end, split_reason in parts:
            part_dur = part_end - part_start
            if part_dur < min_dur_ms:
                continue
            meta = dict(piece["meta"])
            if meta.get("flags"):
                meta["flags"] = list(meta["flags"])
            if split_reason:
                meta["splitBy"] = split_reason
            if part_dur > max_dur_ms:
                part_end = part_start + max_dur_ms
                part_dur = max_dur_ms
                meta["flags"] = meta.get("flags", []) + ["hard_capped"]
                stats["hardCapped"] += 1
            out.append(
                {
                    "start": part_start,
                    "end": part_end,
                    "dur": part_dur,
                    "score": piece["score"],
                    "meta": meta,
                }
            )
    return out, stats


def apply_speech_gate(
    pieces: list[dict],
    audio: np.ndarray,
    sr: int,
    *,
    reject: bool = True,
) -> tuple[list[dict], dict]:
    """Final per-candidate speech-likeness pass (stage 3).

    Stage 1's screen ran on whole VAD regions, which can mix a sentence with
    the clatter right after it. Now that diarization and pause-splitting have
    cut those regions into individual turns, each piece can be judged on its
    own — this is the pass that does the real filtering.

    Pieces below ``SPEECH_SCORE_REJECT`` are dropped (or flagged, when
    ``reject`` is False); pieces between there and ``SPEECH_SCORE_WEAK`` are
    kept but flagged and down-ranked so they sort below confident speech.
    """
    stats = {"speechGateRejected": 0, "speechGateFlagged": 0}
    out: list[dict] = []
    for piece in pieces:
        score, features = _speech_score_for_span(audio, sr, piece["start"], piece["end"])
        meta = dict(piece["meta"])
        if meta.get("flags"):
            meta["flags"] = list(meta["flags"])
        meta["speechScore"] = round(score, 3)
        if features.get("f0_med"):
            meta["f0Med"] = int(round(features["f0_med"]))

        if score < speechlike.SPEECH_SCORE_REJECT:
            if reject:
                stats["speechGateRejected"] += 1
                continue
            meta["flags"] = meta.get("flags", []) + ["possible_non_speech"]
            meta["nonSpeechReason"] = speechlike.describe(features)
            stats["speechGateFlagged"] += 1
            piece = {**piece, "score": min(piece["score"], 0.3)}
        elif score < speechlike.SPEECH_SCORE_WEAK:
            meta["flags"] = meta.get("flags", []) + ["possible_non_speech"]
            meta["nonSpeechReason"] = speechlike.describe(features)
            stats["speechGateFlagged"] += 1
            piece = {**piece, "score": min(piece["score"], 0.5)}

        out.append({**piece, "meta": meta})
    return out, stats


def segments_to_annotations(pieces: list[dict]) -> list[dict]:
    anns: list[dict] = []
    for piece in pieces:
        s = int(round(piece["start"]))
        e = int(round(piece["end"]))
        meta = piece.get("meta") or {}
        ann = {
            "uuid": str(uuid.uuid4()),
            "label": "",
            "startMs": s,
            "endMs": e,
            "tMs": s,
            "source": SOURCE,
            "status": "provisional",
            "score": round(float(piece.get("score", 0.0)), 3),
        }
        if meta.get("splitBy"):
            ann["splitBy"] = meta["splitBy"]
        if meta.get("parentSpanId"):
            ann["parentSpanId"] = meta["parentSpanId"]
        if meta.get("mergedFrom"):
            ann["mergedFrom"] = list(meta["mergedFrom"])
        if meta.get("flags"):
            ann["flags"] = meta["flags"]
        if meta.get("speechScore") is not None:
            ann["speechScore"] = meta["speechScore"]
        if meta.get("nonSpeechReason"):
            ann["nonSpeechReason"] = meta["nonSpeechReason"]
        if meta.get("f0Med"):
            ann["f0Med"] = meta["f0Med"]
        # Anonymous ECAPA clusters stay off the reviewer-facing `speaker`
        # field (Confirm still asks). VTC roles *are* meaningful — prefill
        # Baby/Parent/Other so Confirm can accept with one click.
        if meta.get("speakerCluster"):
            ann["speakerCluster"] = meta["speakerCluster"]
            if meta.get("speakerConfidence") is not None:
                ann["speakerConfidence"] = meta["speakerConfidence"]
        if meta.get("speaker"):
            ann["speaker"] = meta["speaker"]
        if meta.get("vtcRole"):
            ann["vtcRole"] = meta["vtcRole"]
        anns.append(ann)
    return anns


def load_tags(kit: Path) -> list:
    """Read ``tags.json`` for a kit (empty list if missing/absent)."""
    path = kit / "tags.json"
    if not path.exists():
        return []
    tp = load_json(path)
    return tp.get("tags", tp if isinstance(tp, list) else [])


def _tag_interval(tag: dict, *, point_span_ms: float = TAG_POINT_SPAN_MS) -> tuple[float, float] | None:
    """(start_ms, end_ms) a tag occupies, or None if it has no timing at all.

    Tags always carry ``startMs``/``tMs``; ``endMs`` is only set for spans the
    reviewer actually drew — a bare point tag is widened to a short window
    starting at its timestamp (see ``TAG_POINT_SPAN_MS``) so it still blocks
    a new ML candidate from landing right on top of it.
    """
    start = tag.get("startMs")
    if start is None:
        start = tag.get("tMs")
    if start is None:
        return None
    start = float(start)
    end = tag.get("endMs")
    end = float(end) if end is not None else start + point_span_ms
    if end < start:
        start, end = end, start
    return start, end


def _intervals_overlap(
    a_start: float, a_end: float, b_start: float, b_end: float, *, margin_ms: float = 0.0
) -> bool:
    """Standard interval overlap (shares any time), widened by ``margin_ms``
    on each side so near-touching edges still count as an overlap.
    """
    return (a_start - margin_ms) < (b_end + margin_ms) and (b_start - margin_ms) < (a_end + margin_ms)


def filter_tag_overlaps(
    new_anns: list[dict],
    tags: list,
    *,
    margin_ms: float = TAG_OVERLAP_MARGIN_MS,
) -> tuple[list[dict], int]:
    """Drop any new (provisional) candidate that overlaps an existing tag.

    Returns ``(kept, suppressed_count)``. Tags are the source of truth for
    "already reviewed" time — this runs after diarization/pause-splitting so
    it sees final candidate boundaries, not pre-split regions.
    """
    intervals = [iv for iv in (_tag_interval(t) for t in tags) if iv is not None]
    if not intervals:
        return new_anns, 0
    kept: list[dict] = []
    suppressed = 0
    for ann in new_anns:
        start = float(ann.get("startMs", ann.get("tMs", 0)) or 0)
        end = ann.get("endMs")
        end = float(end) if end is not None else start
        if end < start:
            start, end = end, start
        if any(_intervals_overlap(start, end, ts, te, margin_ms=margin_ms) for ts, te in intervals):
            suppressed += 1
            continue
        kept.append(ann)
    return kept, suppressed


def merge_with_existing(
    existing: list,
    new_anns: list[dict],
    *,
    replace_provisional: bool = True,
) -> list:
    """Keep confirmed/dismissed/skipped (and optionally other provisional) intact.

    Skipped means the reviewer parked a candidate for later — preserve it on
    re-run the same way as dismissed (unlike open provisional, which is replaced).
    """
    kept: list = []
    for a in existing:
        status = a.get("status")
        source = a.get("source")
        if status in ("confirmed", "dismissed", "skipped"):
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


def _pieces_from_vtc_turns(turns) -> list[dict]:
    """Convert VTC role turns into pipeline pieces (parent spans)."""
    pieces: list[dict] = []
    for turn in turns:
        pieces.append(
            {
                "start": float(turn.start_ms),
                "end": float(turn.end_ms),
                "dur": float(turn.end_ms - turn.start_ms),
                "score": 0.7,
                "meta": {
                    "speakerCluster": turn.role,
                    "speaker": turn.speaker,
                    "vtcRole": turn.role,
                    "speakerConfidence": round(float(turn.confidence), 3),
                },
            }
        )
    return pieces


def _apply_merge_back_stage(
    pieces: list[dict],
    audio: np.ndarray,
    sr: int,
    *,
    enabled: bool,
    params: reseg_mod.MergeBackParams | None = None,
) -> tuple[list[dict], dict]:
    """Post-DJW short-gated merge-back → word-like children for Review."""
    if not enabled:
        return pieces, {
            "mergeBackEnabled": False,
            "mergeBackIn": len(pieces),
            "mergeBackOut": len(pieces),
            "mergeBackMerges": 0,
        }
    return reseg_mod.merge_back_pieces(
        pieces, audio, sr, params or reseg_mod.MergeBackParams()
    )


def build_candidates_vtc_first(
    audio: np.ndarray,
    sr: int,
    *,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_dur_ms: float = MIN_DUR_MS,
    split_target_ms: float = SPLIT_TARGET_MS,
    reject_non_speech: bool = True,
    resegment: bool = True,
    reseg_target_ms: float = reseg_mod.RESEG_TARGET_MS,
    reseg_min_part_ms: float | None = None,
    word_split_min_sep_ms: float = reseg_mod.WORD_SPLIT_MIN_SEP_MS,
    word_split_min_dip_db: float = reseg_mod.WORD_SPLIT_MIN_DIP_DB,
    merge_back: bool = True,
    merge_back_params: reseg_mod.MergeBackParams | None = None,
) -> tuple[list[dict], dict]:
    """VTC-first path: role timeline → pause/DJW/merge-back → speech gate."""
    times, dbs = frame_rms_db(audio, sr)
    stats: dict = {
        "segmentation": "vtc-first",
        "regions": 0,
        "diarization": {
            "ok": False,
            "backend": "vtc",
            "speakerSplits": 0,
            "numSpeakers": 0,
        },
    }
    try:
        import vtc as vtc_mod
    except ImportError as e:
        stats["diarization"]["error"] = f"vtc.py unavailable: {e}"
        return [], stats

    result = vtc_mod.run_vtc_inference(audio, sr)
    stats["diarization"]["ok"] = result.ok
    stats["diarization"]["backend"] = "vtc"
    if result.cache_hit:
        stats["diarization"]["cacheHit"] = True
    if result.elapsed_sec is not None:
        stats["diarization"]["elapsedSec"] = result.elapsed_sec
    if result.note:
        stats["diarization"]["note"] = result.note
    if result.stats:
        stats["diarization"]["stats"] = result.stats
    if not result.ok:
        stats["diarization"]["error"] = result.error or "VTC failed"
        return [], stats

    # Drop tiny turns the way stage-1 VAD drops short regions.
    turns = [t for t in result.turns if (t.end_ms - t.start_ms) >= min_dur_ms]
    pieces = _pieces_from_vtc_turns(turns)
    stats["regions"] = len(pieces)
    stats["diarization"]["numSpeakers"] = len({t.role for t in turns})
    stats["diarization"]["speakerSplits"] = max(0, len(turns) - 1)

    pieces, refine_stats = refine_long_spans(
        pieces,
        times,
        dbs,
        split_target_ms=split_target_ms,
        min_dur_ms=min_dur_ms,
    )
    stats.update(refine_stats)

    min_part = (
        reseg_min_part_ms
        if reseg_min_part_ms is not None
        else max(min_dur_ms * 0.9, reseg_mod.RESEG_MIN_PART_MS)
    )
    pieces, reseg_stats = reseg_mod.resegment_pieces(
        pieces,
        audio,
        sr,
        enabled=resegment,
        target_ms=reseg_target_ms,
        min_part_ms=min_part,
        word_split_min_sep_ms=word_split_min_sep_ms,
        word_split_min_dip_db=word_split_min_dip_db,
    )
    stats.update(reseg_stats)

    pieces, mb_stats = _apply_merge_back_stage(
        pieces,
        audio,
        sr,
        enabled=bool(resegment and merge_back),
        params=merge_back_params,
    )
    stats.update(mb_stats)

    pieces, gate_stats = apply_speech_gate(pieces, audio, sr, reject=reject_non_speech)
    stats.update(gate_stats)
    stats["split"] = (
        int(stats["diarization"].get("speakerSplits", 0))
        + refine_stats["pauseSplit"]
        + reseg_stats.get("resegSplits", 0)
    )
    stats["candidates"] = len(pieces)
    # merge_gap unused in VTC-first but kept in signature for API symmetry
    _ = merge_gap_ms
    return segments_to_annotations(pieces), stats


def split_by_vtc(
    regions: list[dict],
    audio: np.ndarray,
    sr: int,
    *,
    min_dur_ms: float = MIN_DUR_MS,
) -> tuple[list[dict], dict]:
    """Cut energy-VAD regions at VTC role changes (``diarization=vtc``)."""
    info: dict = {"ok": False, "backend": "vtc", "speakerSplits": 0, "numSpeakers": 0}
    if not regions:
        info["ok"] = True
        return regions, info
    try:
        import vtc as vtc_mod
    except ImportError as e:
        info["error"] = f"vtc.py unavailable: {e}"
        return regions, info

    result = vtc_mod.run_vtc_inference(audio, sr)
    if result.cache_hit:
        info["cacheHit"] = True
    if result.elapsed_sec is not None:
        info["elapsedSec"] = result.elapsed_sec
    if result.stats:
        info["stats"] = result.stats
    if not result.ok:
        info["error"] = result.error or "VTC failed"
        return regions, info

    clipped = vtc_mod.intersect_turns_with_regions(
        result.turns,
        [(r["start"], r["end"]) for r in regions],
        min_ms=min_dur_ms,
    )
    pieces = _pieces_from_vtc_turns(clipped)
    # Preserve stage-1 speech scores where a piece sits inside a region.
    for piece in pieces:
        mid = 0.5 * (piece["start"] + piece["end"])
        for region in regions:
            if region["start"] - 1e-6 <= mid <= region["end"] + 1e-6:
                piece["score"] = region["score"]
                flags = (region.get("meta") or {}).get("flags")
                if flags:
                    piece["meta"]["flags"] = list(flags)
                break
    info["ok"] = True
    info["numSpeakers"] = len({p["meta"]["speakerCluster"] for p in pieces})
    info["speakerSplits"] = max(0, len(pieces) - len(regions))
    return pieces, info


def build_candidates(
    audio: np.ndarray,
    sr: int,
    *,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_dur_ms: float = MIN_DUR_MS,
    speech_delta_db: float = SPEECH_DELTA_DB,
    split_target_ms: float = SPLIT_TARGET_MS,
    reject_non_speech: bool = True,
    diarization: str = "auto",
    num_speakers: int | None = None,
    speaker_distance: float | None = None,
    resegment: bool = True,
    reseg_target_ms: float = reseg_mod.RESEG_TARGET_MS,
    reseg_min_part_ms: float | None = None,
    word_split_min_sep_ms: float = reseg_mod.WORD_SPLIT_MIN_SEP_MS,
    word_split_min_dip_db: float = reseg_mod.WORD_SPLIT_MIN_DIP_DB,
    merge_back: bool = True,
    merge_back_params: reseg_mod.MergeBackParams | None = None,
    segmentation: str = "vad",
) -> tuple[list[dict], dict]:
    """Run VAD → diarize → pause/DJW/merge-back → speech gate; return anns + stats.

    Pass ``segmentation='vtc-first'`` to skip energy VAD/ECAPA and use VTC roles
    as the parent spans instead.
    """
    if (segmentation or "vad").lower() in ("vtc-first", "vtc", "vtc_first"):
        return build_candidates_vtc_first(
            audio,
            sr,
            merge_gap_ms=merge_gap_ms,
            min_dur_ms=min_dur_ms,
            split_target_ms=split_target_ms,
            reject_non_speech=reject_non_speech,
            resegment=resegment,
            reseg_target_ms=reseg_target_ms,
            reseg_min_part_ms=reseg_min_part_ms,
            word_split_min_sep_ms=word_split_min_sep_ms,
            word_split_min_dip_db=word_split_min_dip_db,
            merge_back=merge_back,
            merge_back_params=merge_back_params,
        )

    times, dbs = frame_rms_db(audio, sr)
    regions, stats = detect_speech_regions(
        times,
        dbs,
        audio=audio,
        sr=sr,
        speech_delta_db=speech_delta_db,
        merge_gap_ms=merge_gap_ms,
        min_dur_ms=min_dur_ms,
        reject_non_speech=reject_non_speech,
    )
    stats["segmentation"] = "vad"
    stats["regions"] = len(regions)

    if diarization in (None, "", "none", "off"):
        pieces = regions
        stats["diarization"] = {
            "ok": False,
            "backend": "none",
            "error": "disabled",
            "speakerSplits": 0,
        }
    elif str(diarization).lower() == "vtc":
        pieces, diar_info = split_by_vtc(
            regions, audio, sr, min_dur_ms=min_dur_ms
        )
        stats["diarization"] = diar_info
    else:
        pieces, diar_info = split_by_speaker(
            regions,
            audio,
            sr,
            backend=diarization,
            num_speakers=num_speakers,
            distance=speaker_distance,
        )
        stats["diarization"] = diar_info

    pieces, refine_stats = refine_long_spans(
        pieces,
        times,
        dbs,
        split_target_ms=split_target_ms,
        min_dur_ms=min_dur_ms,
    )
    stats.update(refine_stats)

    min_part = (
        reseg_min_part_ms
        if reseg_min_part_ms is not None
        else max(min_dur_ms * 0.9, reseg_mod.RESEG_MIN_PART_MS)
    )
    pieces, reseg_stats = reseg_mod.resegment_pieces(
        pieces,
        audio,
        sr,
        enabled=resegment,
        target_ms=reseg_target_ms,
        min_part_ms=min_part,
        word_split_min_sep_ms=word_split_min_sep_ms,
        word_split_min_dip_db=word_split_min_dip_db,
    )
    stats.update(reseg_stats)

    pieces, mb_stats = _apply_merge_back_stage(
        pieces,
        audio,
        sr,
        enabled=bool(resegment and merge_back),
        params=merge_back_params,
    )
    stats.update(mb_stats)

    pieces, gate_stats = apply_speech_gate(pieces, audio, sr, reject=reject_non_speech)
    stats.update(gate_stats)
    stats["split"] = (
        int(stats["diarization"].get("speakerSplits", 0))
        + refine_stats["pauseSplit"]
        + reseg_stats.get("resegSplits", 0)
    )
    stats["candidates"] = len(pieces)
    return segments_to_annotations(pieces), stats


def run_vad_on_audio(
    audio_path: Path,
    *,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_dur_ms: float = MIN_DUR_MS,
    speech_delta_db: float = SPEECH_DELTA_DB,
    split_target_ms: float = SPLIT_TARGET_MS,
    reject_non_speech: bool = True,
    diarization: str = "auto",
    num_speakers: int | None = None,
    speaker_distance: float | None = None,
    resegment: bool = True,
    reseg_target_ms: float = reseg_mod.RESEG_TARGET_MS,
    reseg_min_part_ms: float | None = None,
    word_split_min_sep_ms: float = reseg_mod.WORD_SPLIT_MIN_SEP_MS,
    word_split_min_dip_db: float = reseg_mod.WORD_SPLIT_MIN_DIP_DB,
    merge_back: bool = True,
    merge_back_params: reseg_mod.MergeBackParams | None = None,
    segmentation: str = "vad",
) -> tuple[list[dict], dict]:
    audio, sr = sf.read(str(audio_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    return build_candidates(
        audio,
        int(sr),
        merge_gap_ms=merge_gap_ms,
        min_dur_ms=min_dur_ms,
        speech_delta_db=speech_delta_db,
        split_target_ms=split_target_ms,
        reject_non_speech=reject_non_speech,
        diarization=diarization,
        num_speakers=num_speakers,
        speaker_distance=speaker_distance,
        resegment=resegment,
        reseg_target_ms=reseg_target_ms,
        reseg_min_part_ms=reseg_min_part_ms,
        word_split_min_sep_ms=word_split_min_sep_ms,
        word_split_min_dip_db=word_split_min_dip_db,
        merge_back=merge_back,
        merge_back_params=merge_back_params,
        segmentation=segmentation,
    )


def process_kit(
    kit: Path,
    *,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_dur_ms: float = MIN_DUR_MS,
    speech_delta_db: float = SPEECH_DELTA_DB,
    split_target_ms: float = SPLIT_TARGET_MS,
    reject_non_speech: bool = True,
    diarization: str = "auto",
    num_speakers: int | None = None,
    speaker_distance: float | None = None,
    resegment: bool = True,
    reseg_target_ms: float = reseg_mod.RESEG_TARGET_MS,
    merge_back: bool = True,
    merge_back_params: reseg_mod.MergeBackParams | None = None,
    segmentation: str = "vad",
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

    new_anns, vad_stats = run_vad_on_audio(
        audio_path,
        merge_gap_ms=merge_gap_ms,
        min_dur_ms=min_dur_ms,
        speech_delta_db=speech_delta_db,
        split_target_ms=split_target_ms,
        reject_non_speech=reject_non_speech,
        diarization=diarization,
        num_speakers=num_speakers,
        speaker_distance=speaker_distance,
        resegment=resegment,
        reseg_target_ms=reseg_target_ms,
        merge_back=merge_back,
        merge_back_params=merge_back_params,
        segmentation=segmentation,
    )

    tags = load_tags(kit)
    new_anns, tag_overlap_suppressed = filter_tag_overlaps(new_anns, tags)
    vad_stats["tagOverlapSuppressed"] = tag_overlap_suppressed

    existing = []
    ann_path = kit / "annotations.json"
    if ann_path.exists():
        ap = load_json(ann_path)
        existing = ap.get("annotations", ap if isinstance(ap, list) else [])

    merged = merge_with_existing(existing, new_anns, replace_provisional=True)
    if write:
        write_json(ann_path, {"annotations": merged})

    mb = merge_back_params or reseg_mod.MergeBackParams()
    return {
        "ok": True,
        "kit": kit.name,
        "added": len(new_anns),
        "total": len(merged),
        "annotations": new_anns,
        "tagOverlapSuppressed": tag_overlap_suppressed,
        "vadStats": vad_stats,
        "diarization": vad_stats.get("diarization", {}),
        "params": {
            "mergeGapMs": merge_gap_ms,
            "minDurMs": min_dur_ms,
            "speechDeltaDb": speech_delta_db,
            "splitTargetMs": split_target_ms,
            "rejectNonSpeech": reject_non_speech,
            "diarization": diarization,
            "numSpeakers": num_speakers,
            "resegment": resegment,
            "resegTargetMs": reseg_target_ms,
            "mergeBack": bool(resegment and merge_back),
            "mergeBackShortPieceMs": mb.short_piece_ms,
            "mergeBackMaxGapMs": mb.max_gap_ms,
            "mergeBackRequireClearlyShort": mb.require_clearly_short,
            "segmentation": segmentation,
            "source": SOURCE,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", nargs="?", help="Kit folder or library root")
    p.add_argument("--merge-gap-ms", type=float, default=MERGE_GAP_MS)
    p.add_argument("--min-ms", type=float, default=MIN_DUR_MS)
    p.add_argument("--speech-delta-db", type=float, default=SPEECH_DELTA_DB)
    p.add_argument(
        "--split-target-ms",
        type=float,
        default=SPLIT_TARGET_MS,
        help="Same-speaker spans longer than this get a pause re-split (0 disables)",
    )
    p.add_argument(
        "--keep-noise",
        action="store_true",
        help="Disable impulsive non-speech (tap/door) rejection",
    )
    p.add_argument(
        "--segmentation",
        default="vad",
        choices=["vad", "vtc-first"],
        help="Parent-span finder: energy VAD (default) or VTC-first role timeline",
    )
    p.add_argument(
        "--diarization",
        default="auto",
        choices=["auto", "ecapa", "melstats", "pyannote", "vtc", "none"],
        help="Stage-2 speaker diarization backend (default: auto; ignored for vtc-first)",
    )
    p.add_argument(
        "--no-diarization",
        action="store_true",
        help="Skip stage 2 entirely (VAD + pause / syllable splitting only)",
    )
    p.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Force the number of speakers instead of picking it automatically",
    )
    p.add_argument(
        "--speaker-distance",
        type=float,
        default=None,
        help="Cosine-distance cut for speaker clustering (lower = more speakers)",
    )
    p.add_argument(
        "--no-resegment",
        action="store_true",
        help="Skip DJW resegmentation + merge-back (keep coarser spans)",
    )
    p.add_argument(
        "--reseg-target-ms",
        type=float,
        default=reseg_mod.RESEG_TARGET_MS,
        help="Prefer children shorter than this after syllable resegmentation",
    )
    p.add_argument(
        "--no-merge-back",
        action="store_true",
        help="Keep raw DJW children (skip short-gated merge-back to word-like spans)",
    )
    p.add_argument(
        "--merge-back-short-piece-ms",
        type=float,
        default=reseg_mod.MERGE_BACK_SHORT_PIECE_MS,
        help=(
            "Merge-back: min sibling duration under this is clearly short "
            f"(default {reseg_mod.MERGE_BACK_SHORT_PIECE_MS:g})"
        ),
    )
    p.add_argument(
        "--merge-back-max-gap-ms",
        type=float,
        default=reseg_mod.MERGE_BACK_MAX_GAP_MS,
        help=(
            "Merge-back: max gap between siblings to consider merging "
            f"(default {reseg_mod.MERGE_BACK_MAX_GAP_MS:g})"
        ),
    )
    p.add_argument(
        "--list-backends",
        action="store_true",
        help="Print diarization backend availability and exit",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing annotations.json",
    )
    args = p.parse_args(argv)

    if args.list_backends:
        from diarize import backend_status, resolve_backend

        for b in backend_status():
            mark = "available" if b["available"] else "not available"
            print(f"{b['name']:<10} {mark:<15} {b['detail']}")
        try:
            import vtc as vtc_mod

            ok, detail = vtc_mod.vtc_available()
            mark = "available" if ok else "not available"
            print(f"{'vtc':<10} {mark:<15} {detail}")
        except ImportError as e:
            print(f"{'vtc':<10} {'not available':<15} {e}")
        print(f"\nauto would pick: {resolve_backend('auto')}")
        return 0

    if not args.root:
        p.error("the following arguments are required: root")
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

    diarization = "none" if args.no_diarization else args.diarization
    mb_params = reseg_mod.MergeBackParams(
        short_piece_ms=args.merge_back_short_piece_ms,
        max_gap_ms=args.merge_back_max_gap_ms,
    )
    total = 0
    for kit in kits:
        result = process_kit(
            kit,
            merge_gap_ms=args.merge_gap_ms,
            min_dur_ms=args.min_ms,
            speech_delta_db=args.speech_delta_db,
            split_target_ms=args.split_target_ms if args.split_target_ms > 0 else float("inf"),
            reject_non_speech=not args.keep_noise,
            diarization=diarization,
            num_speakers=args.num_speakers,
            speaker_distance=args.speaker_distance,
            resegment=not args.no_resegment,
            reseg_target_ms=args.reseg_target_ms,
            merge_back=not args.no_merge_back,
            merge_back_params=mb_params,
            segmentation=args.segmentation,
            write=not args.dry_run,
        )
        if not result.get("ok"):
            print(f"skip {kit.name}: {result.get('error')}")
            continue
        total += int(result["added"])
        verb = "would write" if args.dry_run else "wrote"
        vs = result.get("vadStats") or {}
        di = result.get("diarization") or {}
        diar_txt = (
            f"{di.get('backend')} · {di.get('numSpeakers', 0)} speaker(s),"
            f" +{di.get('speakerSplits', 0)} speaker splits"
            if di.get("ok")
            else f"diarization off ({di.get('error', 'unavailable')})"
        )
        extra = (
            f" (raw {vs.get('rawRuns', 0)} -> regions {vs.get('regions', 0)}"
            f", screened {vs.get('regionsScreened', 0)} non-speech regions"
            f", +{vs.get('pauseSplit', 0)} pause splits"
            f", +{vs.get('resegSplits', 0)} DJW splits"
            f", {vs.get('mergeBackMerges', 0)} merge-back"
            f", speech gate dropped {vs.get('speechGateRejected', 0)}"
            f" / flagged {vs.get('speechGateFlagged', 0)}"
            f", {vs.get('tagOverlapSuppressed', 0)} already-tagged skipped; {diar_txt})"
        )
        print(f"{kit.name}: {verb} {result['added']} speech segments{extra}")
    print(f"Done. {total} segments across {len(kits)} kit(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
