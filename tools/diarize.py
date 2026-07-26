"""Speaker diarization stage for BabyTalk ML candidates (diar_v1).

This is stage 2 of the ML-candidates pipeline:

    stage 1  vad_segments.py   speech / non-speech  -> speech regions
    stage 2  diarize.py        who is talking       -> speaker-homogeneous turns
    stage 3  vad_segments.py   write annotations    -> ML candidates

Stage 1 only says "someone is making sound here", so a 5-10s region can hold a
parent question, a baby babble, and the parent again. This module cuts those
regions where the *speaker* changes, using real speaker embeddings plus
agglomerative clustering over the whole recording (so the same person keeps the
same label across the session), not an acoustic-shift heuristic.

Backends, in preference order (see ``resolve_backend``):

``pyannote``
    Full ``pyannote.audio`` speaker-diarization pipeline. Best quality
    (its own segmentation, some overlap handling) but needs torch, a
    HuggingFace token, and accepting the model's terms. Opt-in only.
``ecapa``
    SpeechBrain ECAPA-TDNN speaker embeddings (``speechbrain/spkrec-ecapa-voxceleb``,
    ~80 MB, no token) over sliding windows inside the VAD regions, then
    cosine agglomerative clustering. This is the recommended default.
``melstats``
    Pure numpy/sklearn fallback: MFCC mean+std "embeddings" with per-recording
    cepstral mean-variance normalization. No downloads, works out of the box,
    but markedly weaker than ECAPA — it leans on voice timbre/pitch range, so
    it separates adult-vs-baby better than adult-vs-adult.

Every backend returns the same thing: a list of ``Turn`` (start_ms, end_ms,
speaker label, confidence). If none is usable the caller is expected to fall
back to VAD-only candidates rather than fail.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install deps: pip install numpy\n" + str(e)) from e

# Sample rate every speaker model here expects.
TARGET_SR = 16_000
# Sliding window used to look for a speaker change inside one VAD region.
# 1.5s is the usual diarization compromise; shorter windows are noisier
# embeddings, longer ones smear across a turn boundary.
WINDOW_MS = 1_500.0
WINDOW_HOP_MS = 250.0
# Regions shorter than this get one embedding for the whole region (no
# internal search for a speaker change).
MIN_WINDOW_MS = 600.0
# Resolution of the per-region speaker timeline built from window votes.
GRID_MS = 100.0
# Never emit a speaker turn shorter than this — below ~0.5s the embedding
# evidence is too thin to trust a turn boundary.
MIN_TURN_MS = 600.0
# Cosine-distance stopping threshold for agglomerative clustering, per backend.
# ECAPA embeddings of different speakers usually sit well above 0.6 apart;
# the MFCC fallback is centered first (see CENTER_BY_BACKEND) which compresses
# the range, so it wants a lower cut.
DISTANCE_BY_BACKEND = {"ecapa": 0.65, "melstats": 1.0}
# Subtract the recording-mean embedding before clustering? This removes the
# room/mic component that otherwise dominates. Essential for raw MFCC stats,
# harmful for ECAPA (already channel-robust, and centering hurts its
# calibration).
CENTER_BY_BACKEND = {"ecapa": False, "melstats": True}
# Guard rail: refuse to invent more speakers than this in one session.
MAX_SPEAKERS = 8
# A cluster must hold at least this share of the session's windows (and this
# many windows outright) to count as a speaker; smaller ones get absorbed.
MIN_CLUSTER_SHARE = 0.03
MIN_CLUSTER_WINDOWS = 3
# Long sessions can generate tens of thousands of windows; past this we widen
# the hop instead (coarser boundaries, bounded runtime) rather than grinding.
MAX_WINDOWS = 4_000
SPEAKER_PREFIX = "SPEAKER_"

# Keep ECAPA batches small. On CPU torch falls off a fast path somewhere
# between 12 and 16 sequences of this length — measured ~14 ms/clip at batch 8
# versus ~660 ms/clip at batch 16 on an M-series Mac — so a "bigger batch is
# faster" assumption here costs ~50x.
ECAPA_BATCH = 8

MEL_BANDS = 40
MFCC_COEFFS = 20
MFCC_FRAME_MS = 25.0
MFCC_HOP_MS = 10.0


@dataclass
class Turn:
    """One speaker-homogeneous span (absolute ms in the source recording)."""

    start_ms: float
    end_ms: float
    speaker: str
    confidence: float = 1.0


@dataclass
class DiarizationResult:
    turns: list[Turn] = field(default_factory=list)
    backend: str = "none"
    num_speakers: int = 0
    ok: bool = False
    error: str | None = None
    note: str | None = None
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Backend availability
# --------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def hf_token() -> str | None:
    for env in ("BABYTALK_HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN"):
        val = (os.environ.get(env) or "").strip()
        if val:
            return val
    return None


def backend_status() -> list[dict]:
    """Describe each backend so the CLI/UI can explain what's installed."""
    has_torch = _module_available("torch")
    return [
        {
            "name": "pyannote",
            "available": has_torch and _module_available("pyannote.audio") and bool(hf_token()),
            "detail": (
                "pyannote.audio speaker-diarization-3.1 — needs "
                "`pip install pyannote.audio`, a HuggingFace token in "
                "BABYTALK_HF_TOKEN, and accepting the model terms on huggingface.co"
            ),
        },
        {
            "name": "ecapa",
            "available": has_torch and _module_available("speechbrain"),
            "detail": (
                "SpeechBrain ECAPA-TDNN embeddings + agglomerative clustering — "
                "`pip install torch torchaudio speechbrain`, ~80 MB model "
                "downloaded on first run, no token"
            ),
        },
        {
            "name": "melstats",
            "available": True,
            "detail": "numpy MFCC mean/std fallback — always available, weakest quality",
        },
    ]


def resolve_backend(prefer: str | None = None) -> str:
    """Pick a usable backend name. ``prefer='auto'``/None walks the list."""
    status = {b["name"]: b["available"] for b in backend_status()}
    if prefer and prefer not in ("auto", ""):
        if prefer == "none":
            return "none"
        return prefer if status.get(prefer) else "none"
    # pyannote is deliberately not auto-selected: it is much slower on CPU and
    # needs explicit token setup, so the user has to ask for it by name.
    for name in ("ecapa", "melstats"):
        if status.get(name):
            return name
    return "none"


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 at TARGET_SR (speaker models are all 16 kHz)."""
    x = np.asarray(audio)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    if sr == TARGET_SR:
        return x
    try:
        from scipy.signal import resample_poly

        g = math.gcd(int(sr), TARGET_SR)
        return resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    except ImportError:
        n_out = int(round(len(x) * TARGET_SR / float(sr)))
        if n_out <= 1:
            return x[:1]
        idx = np.linspace(0.0, len(x) - 1.0, num=n_out)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def _slice_ms(audio16k: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    s = max(0, int(start_ms * TARGET_SR / 1000.0))
    e = min(len(audio16k), int(end_ms * TARGET_SR / 1000.0))
    if e <= s:
        return np.zeros(1, dtype=np.float32)
    return audio16k[s:e]


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------


class MelStatsEmbedder:
    """MFCC mean+std per window. Real per-window features + clustering, but a
    hand-rolled representation rather than a trained speaker model — treat it
    as the "works with zero downloads" tier.
    """

    name = "melstats"

    def __init__(self) -> None:
        self._fb = _mel_filterbank(TARGET_SR, 512, MEL_BANDS)
        self._dct = _dct2_matrix(MEL_BANDS, MFCC_COEFFS)

    def embed_batch(self, clips: list[np.ndarray]) -> np.ndarray:
        return np.stack([self._embed_one(c) for c in clips], axis=0)

    def _embed_one(self, clip: np.ndarray) -> np.ndarray:
        n_fft = 512
        hop = int(TARGET_SR * MFCC_HOP_MS / 1000.0)
        frame = int(TARGET_SR * MFCC_FRAME_MS / 1000.0)
        x = np.asarray(clip, dtype=np.float64)
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        window = np.hanning(frame)
        n_frames = max(1, (len(x) - frame) // hop + 1)
        idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
        idx = np.clip(idx, 0, len(x) - 1)
        frames = x[idx] * window[None, :]
        spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
        mel = np.log(spec @ self._fb.T + 1e-10)
        mfcc = mel @ self._dct.T  # frames x MFCC_COEFFS
        mfcc = mfcc[:, 1:]  # drop c0 (loudness)
        mean = mfcc.mean(axis=0)
        std = mfcc.std(axis=0)
        if len(mfcc) > 1:
            delta = np.diff(mfcc, axis=0)
            dmean = np.abs(delta).mean(axis=0)
        else:
            dmean = np.zeros_like(mean)
        return np.concatenate([mean, std, dmean]).astype(np.float64)


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN x-vectors (192-d). Downloads ~80 MB on first use."""

    name = "ecapa"

    def __init__(self, model_dir: str | None = None) -> None:
        import torch  # noqa: F401 — import error surfaces to caller
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        savedir = model_dir or os.path.expanduser("~/.cache/babytalk/spkrec-ecapa-voxceleb")
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"},
        )

    def embed_batch(self, clips: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for i in range(0, len(clips), ECAPA_BATCH):
            chunk = clips[i : i + ECAPA_BATCH]
            longest = max(len(c) for c in chunk)
            padded = np.zeros((len(chunk), longest), dtype=np.float32)
            lengths = np.zeros(len(chunk), dtype=np.float32)
            for j, c in enumerate(chunk):
                padded[j, : len(c)] = c
                lengths[j] = len(c) / float(longest)
            with torch.no_grad():
                emb = self._model.encode_batch(
                    torch.from_numpy(padded), torch.from_numpy(lengths)
                )
            out.append(emb.squeeze(1).cpu().numpy().astype(np.float64))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 192))


def load_embedder(backend: str):
    if backend == "ecapa":
        return EcapaEmbedder()
    if backend == "melstats":
        return MelStatsEmbedder()
    raise ValueError(f"no embedder for backend {backend!r}")


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    m_pts = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2)
    hz_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for j in range(left, min(center, fb.shape[1])):
            fb[i, j] = (j - left) / max(1, center - left)
        for j in range(center, min(right, fb.shape[1])):
            fb[i, j] = (right - j) / max(1, right - center)
    return fb


def _dct2_matrix(n_in: int, n_out: int) -> np.ndarray:
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    return np.cos(math.pi * k * (2 * n + 1) / (2.0 * n_in)) * math.sqrt(2.0 / n_in)


# --------------------------------------------------------------------------
# Windowing + clustering + resegmentation
# --------------------------------------------------------------------------


def _region_windows(
    start_ms: float,
    end_ms: float,
    *,
    window_ms: float,
    hop_ms: float,
) -> list[tuple[float, float]]:
    """Sliding windows covering one region; short regions yield one window."""
    dur = end_ms - start_ms
    if dur <= window_ms:
        return [(start_ms, end_ms)]
    wins: list[tuple[float, float]] = []
    t = start_ms
    while t + window_ms <= end_ms + 1e-6:
        wins.append((t, t + window_ms))
        t += hop_ms
    if wins and wins[-1][1] < end_ms - 1e-6:
        wins.append((max(start_ms, end_ms - window_ms), end_ms))
    return wins or [(start_ms, end_ms)]


def _cluster(
    embeddings: np.ndarray,
    *,
    distance: float,
    num_speakers: int | None,
    center: bool,
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) == 0:
        return np.zeros(0, dtype=int)
    if len(embeddings) == 1:
        return np.zeros(1, dtype=int)

    x = np.asarray(embeddings, dtype=np.float64)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
        scale = x.std(axis=0, keepdims=True) + 1e-9
        x = x / scale
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x = x / norms

    if num_speakers and num_speakers > 0:
        # An explicit speaker count is an instruction, not a hint — don't let
        # the tiny-cluster cleanup below quietly undo it.
        k = int(min(num_speakers, len(x)))
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        return model.fit_predict(x)

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance,
    )
    labels = model.fit_predict(x)
    if len(set(labels.tolist())) > MAX_SPEAKERS:
        model = AgglomerativeClustering(
            n_clusters=MAX_SPEAKERS, metric="cosine", linkage="average"
        )
        labels = model.fit_predict(x)
    return _absorb_tiny_clusters(x, labels)


def _absorb_tiny_clusters(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fold near-empty clusters into their nearest real one.

    Threshold-based AHC reliably spits out a handful of one-window clusters —
    a cough, a laugh, a burst of clipping — and each one would otherwise show
    up in the UI as its own "speaker". A cluster has to hold a real share of
    the session before we believe it's a person.
    """
    counts: dict[int, int] = {}
    for lab in labels.tolist():
        counts[lab] = counts.get(lab, 0) + 1
    floor = max(MIN_CLUSTER_WINDOWS, int(math.ceil(MIN_CLUSTER_SHARE * len(labels))))
    big = [lab for lab, c in counts.items() if c >= floor]
    if not big or len(big) == len(counts):
        return labels

    centroids = {lab: x[labels == lab].mean(axis=0) for lab in big}
    keys = list(centroids)
    mat = np.stack([centroids[k] for k in keys], axis=0)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    out = labels.copy()
    for i, lab in enumerate(labels.tolist()):
        if lab in centroids:
            continue
        sims = mat @ x[i]
        out[i] = keys[int(np.argmax(sims))]
    return out


def _rename_by_first_appearance(
    win_labels: np.ndarray, win_times: list[tuple[float, float]]
) -> dict[int, str]:
    """SPEAKER_00 = whoever talks first, so labels are stable and readable."""
    order: list[int] = []
    for _, lab in sorted(zip([w[0] for w in win_times], win_labels.tolist())):
        if lab not in order:
            order.append(lab)
    return {lab: f"{SPEAKER_PREFIX}{i:02d}" for i, lab in enumerate(order)}


def _timeline_labels(
    start_ms: float,
    end_ms: float,
    windows: list[tuple[float, float]],
    labels: list[int],
    *,
    grid_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Majority-vote each grid point over the windows covering it.

    Returns (labels_per_grid_point, vote_fraction_per_grid_point).
    """
    n = max(1, int(math.ceil((end_ms - start_ms) / grid_ms)))
    centers = start_ms + (np.arange(n) + 0.5) * grid_ms
    out = np.zeros(n, dtype=int)
    conf = np.zeros(n, dtype=float)
    uniq = sorted(set(labels))
    lab_index = {lab: i for i, lab in enumerate(uniq)}
    for gi, t in enumerate(centers):
        votes = np.zeros(len(uniq), dtype=float)
        nearest = None
        nearest_d = float("inf")
        for (ws, we), lab in zip(windows, labels):
            wc = 0.5 * (ws + we)
            d = abs(t - wc)
            if d < nearest_d:
                nearest_d, nearest = d, lab
            if ws - 1e-6 <= t <= we + 1e-6:
                # Weight by closeness to the window center: a boundary point
                # belongs more to the window it sits in the middle of.
                votes[lab_index[lab]] += 1.0 / (1.0 + d / 500.0)
        total = float(votes.sum())
        if total <= 0:
            out[gi] = nearest if nearest is not None else uniq[0]
            conf[gi] = 0.5
        else:
            bi = int(np.argmax(votes))
            out[gi] = uniq[bi]
            conf[gi] = float(votes[bi] / total)
    return out, conf


def _runs_from_timeline(
    grid_labels: np.ndarray,
    grid_conf: np.ndarray,
    start_ms: float,
    end_ms: float,
    *,
    grid_ms: float,
    min_turn_ms: float,
) -> list[tuple[float, float, int, float]]:
    """Contiguous same-label runs, with too-short runs absorbed into a neighbor."""
    labels = grid_labels.copy()
    min_pts = max(1, int(round(min_turn_ms / grid_ms)))

    # Absorb short runs into whichever neighbour is longer, repeatedly, until
    # everything left is a defensible turn length.
    for _ in range(8):
        runs = _contiguous_runs(labels)
        if len(runs) <= 1:
            break
        shortest = min(runs, key=lambda r: r[1] - r[0])
        if (shortest[1] - shortest[0]) >= min_pts:
            break
        i = runs.index(shortest)
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i + 1 < len(runs) else None
        if left and right:
            target = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
        else:
            target = left or right
        if target is None:
            break
        labels[shortest[0] : shortest[1]] = target[2]

    out: list[tuple[float, float, int, float]] = []
    for s, e, lab in _contiguous_runs(labels):
        t0 = start_ms + s * grid_ms
        t1 = min(end_ms, start_ms + e * grid_ms)
        if t1 <= t0:
            continue
        out.append((t0, t1, int(lab), float(np.mean(grid_conf[s:e]))))
    if out:
        out[0] = (start_ms, out[0][1], out[0][2], out[0][3])
        last = out[-1]
        out[-1] = (last[0], end_ms, last[2], last[3])
    return out


def _contiguous_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if len(labels) == 0:
        return runs
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, int(labels[start])))
            start = i
    return runs


def diarize_regions(
    audio: np.ndarray,
    sr: int,
    regions: list[tuple[float, float]],
    *,
    backend: str = "auto",
    num_speakers: int | None = None,
    distance: float | None = None,
    window_ms: float = WINDOW_MS,
    hop_ms: float = WINDOW_HOP_MS,
    min_turn_ms: float = MIN_TURN_MS,
    embedder=None,
) -> DiarizationResult:
    """Split VAD ``regions`` (ms pairs) into speaker-homogeneous turns.

    Embeddings are clustered across the *whole recording* so SPEAKER_00 means
    the same person everywhere, then each region is re-segmented at the points
    where the winning cluster changes.
    """
    name = resolve_backend(backend)
    if name == "none":
        return DiarizationResult(
            backend="none",
            ok=False,
            error="No diarization backend available",
        )
    if name == "pyannote":
        return _diarize_pyannote(
            audio, sr, regions, num_speakers=num_speakers, min_turn_ms=min_turn_ms
        )

    if not regions:
        return DiarizationResult(backend=name, ok=True, turns=[], num_speakers=0)

    try:
        emb = embedder if embedder is not None else load_embedder(name)
    except Exception as e:  # noqa: BLE001 — missing model/deps must degrade, not crash
        return DiarizationResult(backend=name, ok=False, error=f"{type(e).__name__}: {e}")

    audio16k = to_mono_16k(audio, sr)
    total_ms = len(audio16k) * 1000.0 / TARGET_SR

    speech_ms = sum(max(0.0, e - s) for s, e in regions)
    est_windows = speech_ms / max(hop_ms, 1.0)
    if est_windows > MAX_WINDOWS:
        hop_ms = hop_ms * est_windows / MAX_WINDOWS

    all_windows: list[tuple[float, float]] = []
    owner: list[int] = []
    for ri, (rs, re_) in enumerate(regions):
        rs = max(0.0, rs)
        re_ = min(total_ms, re_)
        if re_ - rs < MIN_WINDOW_MS / 2:
            all_windows.append((rs, re_))
            owner.append(ri)
            continue
        for w in _region_windows(rs, re_, window_ms=window_ms, hop_ms=hop_ms):
            all_windows.append(w)
            owner.append(ri)

    clips = [_slice_ms(audio16k, s, e) for s, e in all_windows]
    if not clips:
        return DiarizationResult(backend=name, ok=True, turns=[], num_speakers=0)

    try:
        embeddings = emb.embed_batch(clips)
    except Exception as e:  # noqa: BLE001
        return DiarizationResult(backend=name, ok=False, error=f"{type(e).__name__}: {e}")

    dist = distance if distance is not None else DISTANCE_BY_BACKEND.get(name, 0.6)
    labels = _cluster(
        embeddings,
        distance=dist,
        num_speakers=num_speakers,
        center=CENTER_BY_BACKEND.get(name, True),
    )
    naming = _rename_by_first_appearance(labels, all_windows)

    turns: list[Turn] = []
    split_count = 0
    for ri, (rs, re_) in enumerate(regions):
        idxs = [i for i, o in enumerate(owner) if o == ri]
        if not idxs:
            continue
        win = [all_windows[i] for i in idxs]
        lab = [int(labels[i]) for i in idxs]
        if len(set(lab)) == 1:
            turns.append(Turn(rs, re_, naming[lab[0]], 1.0))
            continue
        grid_labels, grid_conf = _timeline_labels(rs, re_, win, lab, grid_ms=GRID_MS)
        runs = _runs_from_timeline(
            grid_labels, grid_conf, rs, re_, grid_ms=GRID_MS, min_turn_ms=min_turn_ms
        )
        if len(runs) > 1:
            split_count += len(runs) - 1
        for t0, t1, l, c in runs:
            turns.append(Turn(t0, t1, naming[l], round(c, 3)))

    speakers = sorted({t.speaker for t in turns})
    return DiarizationResult(
        turns=turns,
        backend=name,
        num_speakers=len(speakers),
        ok=True,
        stats={
            "windows": len(all_windows),
            "regions": len(regions),
            "speakerSplits": split_count,
            "distance": dist,
            "speakers": speakers,
        },
    )


def _diarize_pyannote(
    audio: np.ndarray,
    sr: int,
    regions: list[tuple[float, float]],
    *,
    num_speakers: int | None,
    min_turn_ms: float,
) -> DiarizationResult:
    """Run the pyannote pipeline on the whole file, then keep only the parts
    that overlap the VAD regions (so stage 1's non-speech filtering still
    applies and candidate boundaries stay comparable across backends).
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        return DiarizationResult(backend="pyannote", ok=False, error=str(e))

    token = hf_token()
    if not token:
        return DiarizationResult(
            backend="pyannote",
            ok=False,
            error="Set BABYTALK_HF_TOKEN (and accept pyannote/speaker-diarization-3.1 terms)",
        )
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        audio16k = to_mono_16k(audio, sr)
        waveform = torch.from_numpy(audio16k).unsqueeze(0)
        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        diarization = pipeline({"waveform": waveform, "sample_rate": TARGET_SR}, **kwargs)
    except Exception as e:  # noqa: BLE001
        return DiarizationResult(backend="pyannote", ok=False, error=f"{type(e).__name__}: {e}")

    raw: list[Turn] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        raw.append(Turn(segment.start * 1000.0, segment.end * 1000.0, str(speaker), 1.0))
    raw.sort(key=lambda t: t.start_ms)

    turns: list[Turn] = []
    split_count = 0
    for rs, re_ in regions:
        pieces = []
        for t in raw:
            s, e = max(rs, t.start_ms), min(re_, t.end_ms)
            if e - s >= min_turn_ms / 2:
                pieces.append(Turn(s, e, t.speaker, 1.0))
        if not pieces:
            turns.append(Turn(rs, re_, "", 0.0))
            continue
        pieces.sort(key=lambda t: t.start_ms)
        # Stretch to cover the whole region so no audio is silently dropped.
        pieces[0].start_ms = rs
        pieces[-1].end_ms = re_
        for i in range(len(pieces) - 1):
            mid = 0.5 * (pieces[i].end_ms + pieces[i + 1].start_ms)
            pieces[i].end_ms = mid
            pieces[i + 1].start_ms = mid
        split_count += len(pieces) - 1
        turns.extend(pieces)

    speakers = sorted({t.speaker for t in turns if t.speaker})
    return DiarizationResult(
        turns=turns,
        backend="pyannote",
        num_speakers=len(speakers),
        ok=True,
        stats={
            "regions": len(regions),
            "speakerSplits": split_count,
            "speakers": speakers,
        },
    )
