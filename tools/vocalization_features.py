"""Per-vocalization feature extraction (duration, DJW syllable count, pitch).

Used by Find speech in ``unit=vocalization`` mode: DJW nuclei are counted
inside each pause/diarization window — they do **not** define segment bounds.
"""

from __future__ import annotations

import numpy as np

from resegment import (
    VOICING_PEAK,
    _local_maxima,
    _praat_like_intensity,
    _voiced_at_peaks,
    find_djw_nuclei,
)
from speechlike import HOP_MS, _frames, _to_mono_analysis_sr, _voicing


def syllable_count(samples: np.ndarray, sr: int) -> int:
    """Approximate syllable count via DJW voiced intensity nuclei (count only)."""
    if len(samples) < 16:
        return 0
    times_ms, intensity = _praat_like_intensity(samples, sr)
    if len(intensity) < 3:
        return 1 if len(samples) > int(0.08 * sr) else 0
    raw_peaks = _local_maxima(intensity)
    voiced = None
    if raw_peaks:
        voiced_list = _voiced_at_peaks(samples, sr, times_ms, raw_peaks)
        voiced = [False] * len(intensity)
        for idx, flag in zip(raw_peaks, voiced_list):
            if 0 <= idx < len(voiced):
                voiced[idx] = bool(flag)
    nuclei = find_djw_nuclei(times_ms, intensity, voiced)
    if len(samples) > int(0.05 * sr):
        return max(1, len(nuclei))
    return len(nuclei)


def pitch_contour_features(samples: np.ndarray, sr: int) -> dict:
    """Pitch contour shape / variability for stage labeling."""
    empty = {
        "f0Med": 0.0,
        "f0Mean": 0.0,
        "f0Std": 0.0,
        "f0Range": 0.0,
        "f0Slope": 0.0,
        "voicedFraction": 0.0,
        "pitchVariability": 0.0,
        "contourComplexity": 0.0,
    }
    x, asr = _to_mono_analysis_sr(samples, sr)
    if len(x) < 32:
        return empty

    frames = _frames(x, asr)
    rms = np.sqrt(np.mean(frames * frames, axis=1)) + 1e-12
    env_db = 20.0 * np.log10(rms)
    active = env_db >= float(env_db.max()) - 22.0
    if not active.any():
        active = np.ones_like(env_db, dtype=bool)

    voicing_peak, f0 = _voicing(frames, asr)
    voiced_mask = (voicing_peak >= VOICING_PEAK) & active
    f0_v = f0[voiced_mask]
    voiced_fraction = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0

    if len(f0_v) < 2:
        empty["voicedFraction"] = voiced_fraction
        if len(f0_v) == 1:
            empty["f0Med"] = float(f0_v[0])
            empty["f0Mean"] = float(f0_v[0])
        return empty

    f0_med = float(np.median(f0_v))
    f0_mean = float(np.mean(f0_v))
    f0_std = float(np.std(f0_v))
    f0_range = float(np.max(f0_v) - np.min(f0_v))
    t = np.arange(len(f0))[voiced_mask].astype(np.float64) * (HOP_MS / 1000.0)
    if len(t) >= 2 and float(np.std(t)) > 1e-6:
        slope = float(np.polyfit(t, f0_v.astype(np.float64), 1)[0])
    else:
        slope = 0.0
    pitch_var = float(f0_std / max(f0_med, 1.0))

    if len(f0_v) >= 4:
        smooth = np.convolve(f0_v, np.ones(3) / 3.0, mode="valid")
        d = np.diff(smooth)
        sign = np.sign(d)
        sign = sign[sign != 0]
        flips = int(np.sum(sign[1:] * sign[:-1] < 0)) if len(sign) > 1 else 0
        contour_complexity = float(flips / max(len(smooth) - 1, 1))
    else:
        contour_complexity = 0.0

    return {
        "f0Med": f0_med,
        "f0Mean": f0_mean,
        "f0Std": f0_std,
        "f0Range": f0_range,
        "f0Slope": abs(slope),
        "voicedFraction": voiced_fraction,
        "pitchVariability": pitch_var,
        "contourComplexity": contour_complexity,
    }


def extract_span_features(
    audio: np.ndarray, sr: int, start_ms: float, end_ms: float
) -> dict:
    """Features for one vocalization window."""
    s = max(0, int(start_ms * sr / 1000.0))
    e = min(len(audio), int(end_ms * sr / 1000.0))
    if e <= s:
        e = min(len(audio), s + int(0.05 * sr))
    samples = np.asarray(audio[s:e], dtype=np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration_ms = max(0.0, float(end_ms) - float(start_ms))
    syll = syllable_count(samples, sr)
    pitch = pitch_contour_features(samples, sr)
    rate = (syll / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0
    out = {
        "durationMs": round(duration_ms, 1),
        "syllableCount": int(syll),
        "syllableRate": round(rate, 3),
    }
    for k, v in pitch.items():
        out[k] = round(float(v), 4) if isinstance(v, float) else v
    return out
