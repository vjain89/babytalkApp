"""Absolute "does this sound like a voice?" scoring for VAD candidates.

The energy VAD in ``vad_segments.py`` answers *"is something louder than the
room here?"*, which is equally true of a tap running, a door closing, a chair
scraping and a spoon hitting a bowl. This module answers the different
question — *"does this look like it came out of a vocal tract?"* — from three
classical, model-free cues that survive cheap phone mics:

``voicing_peak_med`` / ``voiced_fraction``
    Strength and prevalence of a periodic peak in the normalized
    autocorrelation at a plausible f0 (adult male floor through infant
    squeal). Vowels and babble are periodic; impacts and running water are not.
``speech_band_ratio``
    Share of energy in ~300–3400 Hz, where voices live.
``low_band_ratio``
    Share of energy below 250 Hz. Thumps, handling noise, footsteps and door
    closes are dominated by rumble that voices don't produce.

Why absolute and not relative: the previous check z-scored each segment
against the others in the same recording, so it could only ever flag a fixed
slice of every session as "the most noise-like" — a recording that was 90%
kitchen noise still yielded 90% kitchen-noise candidates, and a clean
recording still had ~15% of its real speech flagged. An absolute judgement can
say "this whole session is junk" or "none of this is junk".

Calibration (see ``tools/README.md``): scored against 457 spans a reviewer had
already confirmed (206) or dismissed (251) across their own sessions. Combined
AUC 0.81, and stable per-kit (0.77 / 0.83 / 0.88). Fitting logistic weights
instead of using the fixed ones below was tried and did *not* generalize
better under leave-one-kit-out CV (best 0.79), so the weights here are fixed
and interpretable rather than learned. A syllabic-modulation (2–10 Hz envelope
rhythm) feature was also tried and discarded — textbook speech cue, but it did
not separate this data at all (AUC 0.46), most likely because these segments
are single short utterances rather than continuous speech.

Deliberately dependency-free beyond numpy, so stage 1 keeps working when torch
and speechbrain are absent.
"""

from __future__ import annotations

import numpy as np

# Analysis rate. Callers hand us arbitrary phone sample rates; we decimate to
# keep the FFTs small since every feature here is a coarse statistic.
ANALYSIS_SR = 16_000

FRAME_MS = 40.0
HOP_MS = 10.0

# f0 search range: adult male floor (~70 Hz) up through infant squeals
# (~600 Hz). Wider than the adult-speech convention on purpose — this codebase
# is mostly babies.
F0_MIN_HZ = 70.0
F0_MAX_HZ = 600.0
# Normalized-autocorrelation peak above which a frame counts as voiced.
VOICING_PEAK = 0.42
# Only judge voicing on frames carrying the segment's energy: quiet tails are
# unvoiced by definition and would dilute the fraction.
ACTIVE_FRAME_DB_BELOW_PEAK = 22.0

SPEECH_BAND_HZ = (300.0, 3400.0)
LOW_BAND_HZ = 250.0

# Fixed score weights (see calibration note in the module docstring).
W_SPEECH_BAND = 0.40
W_VOICING_PEAK = 0.25
W_VOICED_FRACTION = 0.15
W_NOT_LOW_BAND = 0.20

# Below this a candidate is dropped as non-speech. Chosen to keep ~93% of
# confirmed speech while removing ~52% of dismissed junk.
SPEECH_SCORE_REJECT = 0.55
# Between REJECT and WEAK: keep it, but flag it and lower its score so it
# sorts down — the reviewer decides. Above WEAK we say nothing.
SPEECH_SCORE_WEAK = 0.68
# Pre-diarization screen on whole VAD regions. Deliberately far more lenient
# than the per-candidate gate: a region can hold speech *and* noise, and it
# gets re-judged piece by piece after diarization anyway. This pass exists to
# keep obvious junk out of the speaker clustering (and off the ECAPA bill).
SPEECH_SCORE_REGION_REJECT = 0.42


def _to_mono_analysis_sr(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr <= ANALYSIS_SR or len(x) < 8:
        return x, int(sr)
    # Integer decimation with a light anti-alias moving average.
    factor = max(1, int(round(sr / ANALYSIS_SR)))
    if factor <= 1:
        return x, int(sr)
    x = np.convolve(x, np.ones(factor) / factor, mode="same")[::factor]
    return x, int(round(sr / factor))


def _frames(x: np.ndarray, sr: int) -> np.ndarray:
    frame = max(16, int(sr * FRAME_MS / 1000.0))
    hop = max(1, int(sr * HOP_MS / 1000.0))
    if len(x) < frame:
        padded = np.zeros(frame, dtype=np.float64)
        padded[: len(x)] = x
        return padded[None, :]
    n = (len(x) - frame) // hop + 1
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _voicing(frames: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (normalized autocorrelation peak, f0 estimate in Hz)."""
    n_frames, frame = frames.shape
    # Mean-remove so a DC offset can't fake perfect periodicity.
    f = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1
    while nfft < 2 * frame:
        nfft *= 2
    spec = np.fft.rfft(f, n=nfft, axis=1)
    acf = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :frame]
    zero_lag = acf[:, :1].copy()
    zero_lag[zero_lag <= 0] = 1e-12
    acf = acf / zero_lag

    min_lag = max(2, int(sr / F0_MAX_HZ))
    max_lag = min(frame - 1, int(sr / F0_MIN_HZ))
    if max_lag <= min_lag:
        return np.zeros(n_frames), np.zeros(n_frames)
    band = acf[:, min_lag : max_lag + 1]
    best = np.argmax(band, axis=1)
    peak = band[np.arange(n_frames), best]
    f0 = sr / np.maximum(best + min_lag, 1)
    return np.clip(peak, 0.0, 1.0), f0


def _band_ratios(frames: np.ndarray, sr: int, weights: np.ndarray) -> dict:
    frame = frames.shape[1]
    window = np.hanning(frame) if frame > 1 else np.ones(frame)
    spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)

    total = spec.sum(axis=1) + 1e-12
    speech = spec[:, (freqs >= SPEECH_BAND_HZ[0]) & (freqs <= SPEECH_BAND_HZ[1])].sum(axis=1)
    low = spec[:, freqs < LOW_BAND_HZ].sum(axis=1)

    # Loudness-weighted so a long quiet tail can't outvote the actual event.
    w = weights / (weights.sum() + 1e-12)
    return {
        "speech_band_ratio": float(np.sum(w * (speech / total))),
        "low_band_ratio": float(np.sum(w * (low / total))),
    }


_EMPTY = {
    "voiced_fraction": 0.0,
    "voicing_peak_med": 0.0,
    "f0_med": 0.0,
    "speech_band_ratio": 0.0,
    "low_band_ratio": 1.0,
}


def speech_likeness_features(audio: np.ndarray, sr: int) -> dict:
    """Raw per-segment features. See the module docstring for what each means."""
    x, asr = _to_mono_analysis_sr(audio, sr)
    if len(x) < 32:
        return dict(_EMPTY)

    frames = _frames(x, asr)
    rms = np.sqrt(np.mean(frames * frames, axis=1)) + 1e-12
    env_db = 20.0 * np.log10(rms)

    active = env_db >= float(env_db.max()) - ACTIVE_FRAME_DB_BELOW_PEAK
    if not active.any():
        active = np.ones_like(env_db, dtype=bool)

    voicing_peak, f0 = _voicing(frames, asr)
    vp_active = voicing_peak[active]
    voiced = vp_active >= VOICING_PEAK
    f0_voiced = f0[active][voiced] if voiced.any() else np.zeros(0)

    out = {
        "voiced_fraction": float(np.mean(voiced)) if len(vp_active) else 0.0,
        "voicing_peak_med": float(np.median(vp_active)) if len(vp_active) else 0.0,
        "f0_med": float(np.median(f0_voiced)) if len(f0_voiced) else 0.0,
    }
    out.update(_band_ratios(frames, asr, weights=rms * active))
    return out


def speech_score(features: dict) -> float:
    """Collapse the features into one 0–1 "this is a voice" score."""
    return float(
        np.clip(
            W_SPEECH_BAND * features.get("speech_band_ratio", 0.0)
            + W_VOICING_PEAK * features.get("voicing_peak_med", 0.0)
            + W_VOICED_FRACTION * features.get("voiced_fraction", 0.0)
            + W_NOT_LOW_BAND * (1.0 - features.get("low_band_ratio", 1.0)),
            0.0,
            1.0,
        )
    )


def speech_likeness(audio: np.ndarray, sr: int) -> tuple[float, dict]:
    """``(score, features)`` for one audio segment."""
    features = speech_likeness_features(audio, sr)
    return speech_score(features), features


def describe(features: dict) -> str:
    """Short human-readable reason a segment scored the way it did, for the UI.

    Always returns something: a flagged candidate with no explanation next to
    it is worse than a vague one, because the reviewer can't tell whether the
    flag means anything.
    """
    reasons = []
    if features.get("low_band_ratio", 0.0) >= 0.45:
        reasons.append("low-frequency rumble")
    if features.get("speech_band_ratio", 1.0) < 0.45:
        reasons.append("little energy in the voice band")
    if features.get("voicing_peak_med", 1.0) < 0.40:
        reasons.append("not periodic/voiced")
    if reasons:
        return ", ".join(reasons)
    # Nothing individually bad, just weak across the board.
    return "weakly voice-like overall"
