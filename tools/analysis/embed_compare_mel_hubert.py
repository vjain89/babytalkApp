#!/usr/bin/env python3
"""Compare mel fingerprints vs HuBERT embeddings on hand-annotated tags.

Loads ``tags.json`` (hand labels, not ML candidates) from two BabyTalk kits,
embeds each labeled clip with:

  * **mel** — ``cluster_sounds.log_mel_embed`` (production fingerprint), or an
    analysis-only **normalized** variant (RMS loudness + drop duration feature;
    still resampled to a fixed mel time grid)
  * **HuBERT** — ``facebook/hubert-base-ls960`` mean-pooled hidden states
    (first run downloads ~360 MB from HuggingFace)

Then reports same-label vs different-label pairwise **cosine distances**
(``sklearn.metrics.pairwise.cosine_distances`` = 1 − cosine similarity),
histogram + UMAP plots, and a summary gap table.

Label key
---------
Prefer ``word`` (lowercased, stripped). If empty/missing, fall back to
``label``, then ``phonetic``. Skip empty and ``untitled``.

Non-verbal exclusion
--------------------
``--exclude-nonverbal`` drops tags whose ``category`` is
``non-verbal vocalization`` (case-insensitive) before embedding / distances.
Matches the structured taxonomy field used by the review UI (not the joined
``label`` string alone).

Mel normalization (analysis-only)
---------------------------------
Production ``log_mel_embed`` already peak-normalizes and interpolates to a
fixed frame grid, but **appends log10(duration_ms)** as an extra feature.
``--mel-norm`` uses an analysis wrapper that:

  1. **Loudness:** RMS-normalize each clip to a fixed target RMS before STFT
  2. **Duration:** keep the fixed-frame mel grid (duration-invariant shape)
     and **omit** the duration scalar so length cannot dominate distance

Does not change ``cluster_sounds.py`` / production clustering.

Kit resolution
--------------
Match needles against folder name, ``sessionName``, ``originalSessionName``,
``title``, and ``displayName`` in ``manifest.json``. Default library:
``~/Documents/BabyTalk/Library`` (via ``babytalk_paths``).

Requirements
------------
  tools/.venv/bin/pip install numpy soundfile scikit-learn matplotlib \\
      transformers torch umap-learn

Usage
-----
  # Full condition matrix (recommended):
  tools/.venv/bin/python tools/analysis/embed_compare_mel_hubert.py --matrix \\
      --out tools/analysis/out/embed_compare_norm

  # Reuse embeddings_cache.npz; force recompute with --refresh-cache.

  # Single condition:
  tools/.venv/bin/python tools/analysis/embed_compare_mel_hubert.py \\
      --mel-norm --exclude-nonverbal

Shared-scale deliverables (matrix mode)
---------------------------------------
  hist_mel_vs_hubert_{all,excl_nonverbal}.png — mel(norm on)|HuBERT, density,
      shared xlim [0,2] and shared ylim
  umap_mel_vs_hubert_{all,excl_nonverbal}.png — side-by-side UMAP with shared
      xlim/ylim (padded union of both fits)

LDA supervised projection (``--lda``)
-------------------------------------
Fit a linear discriminant on **mel norm-on** excl_nonverbal embeddings
(label = word key) to maximize same- vs different-label separation, then
compare pairwise cosine gaps + UMAP vs raw mel on the same clip set.

  tools/.venv/bin/python tools/analysis/embed_compare_mel_hubert.py --lda \\
      --out tools/analysis/out/embed_compare_norm --skip-hubert

Train protocol: fit-on-all labeled excl_nonverbal clips (optimistic diagnostic
upper bound — not leave-one-kit-out). Prefers sklearn LDA to
``min(n_classes-1, n_features, 32)`` dims; falls back to PCA+LDA or NCA.
Rows L2-normalized after transform for cosine distance.

Leave-one-kit-out generalization (``--lda-loko``)
-------------------------------------------------
Train LDA on all excl_nonverbal clips from the other kit(s), transform the
held-out kit, and score cosine gap **only on held-out pairs**. Also reports
raw mel gap on that kit and compares to fit-on-all LDA. Optionally holds out
~20% of multi-instance label types (seeded) and scores those clips.

  tools/.venv/bin/python tools/analysis/embed_compare_mel_hubert.py --lda-loko \\
      --out tools/analysis/out/embed_compare_norm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from babytalk_paths import LIBRARY_DIR, list_local_kits  # noqa: E402
from cluster_sounds import (  # noqa: E402
    HOP,
    N_FFT,
    N_MELS,
    _mel_filterbank,
    log_mel_embed,
)

OUT_DIR = Path(__file__).resolve().parent / "out" / "embed_compare"
OUT_DIR_MATRIX = Path(__file__).resolve().parent / "out" / "embed_compare_norm"
LDA_MAX_COMPONENTS = 32

DEFAULT_NEEDLES = (
    "26_07_27__19:53:00",
    "26_07_05__00:00:00",
)

# Cosine distance on ~N=350 clips is tractable (~60k pairs). Cap different-
# label pairs for histograms only if N grows large.
MAX_DIFF_PAIRS_HIST = 50_000
HUBERT_SR = 16_000
HUBERT_MODEL_ID = "facebook/hubert-base-ls960"
MIN_SAMPLES = 32  # skip absurdly short clips after slicing
NONVERBAL_CATEGORY = "non-verbal vocalization"
TARGET_RMS = 0.1
N_FRAMES_OUT = 32


def _require_deps(*, need_hubert: bool, need_umap: bool) -> None:
    missing = []
    for mod, pip in (
        ("soundfile", "soundfile"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip)
    if need_hubert:
        for mod, pip in (("torch", "torch"), ("transformers", "transformers")):
            try:
                __import__(mod)
            except ImportError:
                missing.append(pip)
    if need_umap:
        try:
            __import__("umap")
        except ImportError:
            # Soft: continue without UMAP; main returns non-zero after plots.
            print(
                "[umap] umap-learn not installed — UMAP plots will be skipped.\n"
                "  tools/.venv/bin/pip install umap-learn",
                flush=True,
            )
    if missing:
        pkgs = " ".join(dict.fromkeys(missing))
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall with:\n"
            f"  tools/.venv/bin/pip install {pkgs}\n"
        )


def label_key(tag: dict) -> str | None:
    """Same-label key: word → label → phonetic (lowercased/stripped)."""
    for field in ("word", "label", "phonetic"):
        raw = tag.get(field)
        if raw is None:
            continue
        val = str(raw).strip().lower()
        if val and val != "untitled":
            return val
    return None


def tag_category(tag: dict) -> str:
    return str(tag.get("category") or "").strip().lower()


def is_nonverbal(tag: dict) -> bool:
    """True if taxonomy ``category`` is the catch-all non-verbal vocalization."""
    return tag_category(tag) == NONVERBAL_CATEGORY


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
    kits = list_local_kits(library)
    found: list[tuple[str, Path]] = []
    for needle in needles:
        matches = [k for k in kits if needle in kit_search_blob(k)]
        if not matches:
            raise SystemExit(
                f"No kit matching {needle!r} under {library}\n"
                f"(searched folder name + sessionName/originalSessionName/title/displayName)"
            )
        if len(matches) > 1:
            names = ", ".join(m.name for m in matches)
            print(f"[warn] multiple kits match {needle!r}: {names}; using {matches[0].name}")
        found.append((needle, matches[0]))
    return found


def load_hand_tags(kit: Path) -> list[dict]:
    path = kit / "tags.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    tags = data.get("tags", data if isinstance(data, list) else [])
    out = []
    for t in tags:
        start = t.get("startMs")
        end = t.get("endMs")
        if start is None or end is None:
            continue
        start_f, end_f = float(start), float(end)
        if end_f <= start_f:
            continue
        key = label_key(t)
        if not key:
            continue
        out.append(
            {
                **t,
                "_start": start_f,
                "_end": end_f,
                "_label": key,
                "_category": tag_category(t),
                "_nonverbal": is_nonverbal(t),
            }
        )
    return out


def resolve_audio(kit: Path) -> Path:
    man = {}
    man_path = kit / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
    return kit / man.get("audioFile", "audio.wav")


def slice_samples(
    audio: np.ndarray, sr: int, start_ms: float, end_ms: float
) -> np.ndarray | None:
    start = max(0, int(start_ms * sr / 1000.0))
    end = min(len(audio), int(end_ms * sr / 1000.0))
    if end <= start:
        return None
    clip = np.asarray(audio[start:end], dtype=np.float64)
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    if len(clip) < MIN_SAMPLES:
        return None
    return clip


def rms_normalize(samples: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """Scale waveform to a fixed RMS (analysis-only loudness normalization)."""
    x = np.asarray(samples, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x))) + 1e-12
    return x * (target_rms / rms)


def log_mel_embed_analysis(
    samples: np.ndarray,
    sr: int,
    *,
    normalize: bool,
    target_rms: float = TARGET_RMS,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop: int = HOP,
    n_frames_out: int = N_FRAMES_OUT,
) -> np.ndarray:
    """Mel fingerprint for analysis.

    ``normalize=False`` → production ``log_mel_embed`` (peak norm + fixed
    frame grid + log-duration feature).

    ``normalize=True`` → analysis-only path:
      * RMS loudness normalize to ``target_rms``
      * same STFT → mel → interpolate to ``n_frames_out`` frames
      * per-bin z-score across time
      * **no** duration scalar (duration-invariant fingerprint)
      * L2-normalize
    """
    if not normalize:
        return log_mel_embed(
            samples,
            sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop=hop,
            n_frames_out=n_frames_out,
        )

    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = rms_normalize(np.asarray(samples, dtype=np.float64), target_rms)
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))
    window = np.hanning(n_fft)
    fb = _mel_filterbank(sr, n_fft, n_mels)
    frames = []
    for start in range(0, max(1, len(samples) - n_fft + 1), hop):
        frame = samples[start : start + n_fft] * window
        spec = np.abs(np.fft.rfft(frame)) ** 2
        mel = fb @ spec
        frames.append(np.log(mel + 1e-10))
    dim = n_mels * n_frames_out
    if not frames:
        return np.zeros(dim, dtype=np.float64)
    mat = np.stack(frames, axis=0)  # T x n_mels
    t_old = np.linspace(0.0, 1.0, num=len(mat))
    t_new = np.linspace(0.0, 1.0, num=n_frames_out)
    fixed = np.zeros((n_frames_out, n_mels), dtype=np.float64)
    for j in range(n_mels):
        fixed[:, j] = np.interp(t_new, t_old, mat[:, j])
    fixed = fixed - fixed.mean(axis=0, keepdims=True)
    std = fixed.std(axis=0, keepdims=True) + 1e-8
    fixed = fixed / std
    feat = fixed.reshape(-1)
    # Intentionally omit production's log10(duration_ms) feature.
    norm = np.linalg.norm(feat) + 1e-12
    return (feat / norm).astype(np.float64)


def resample_to(samples: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return samples.astype(np.float32)
    import torch
    import torchaudio

    wav = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
    out = torchaudio.functional.resample(wav, sr, target_sr)
    return out.squeeze(0).numpy()


def hubert_embed_batch(
    clips_16k: list[np.ndarray],
    *,
    batch_size: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """Mean-pool last hidden state over time → (N, hidden)."""
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    print(
        f"[hubert] loading {HUBERT_MODEL_ID} "
        "(downloads ~360 MB on first run)…",
        flush=True,
    )
    # Prefer local HF cache (offline) so sandbox/proxy cannot block reloads.
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            HUBERT_MODEL_ID, local_files_only=True
        )
        model = HubertModel.from_pretrained(HUBERT_MODEL_ID, local_files_only=True)
    except (OSError, ValueError):
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_ID)
        model = HubertModel.from_pretrained(HUBERT_MODEL_ID)
    model.eval()
    model.to(device)

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(clips_16k), batch_size):
            batch = clips_16k[i : i + batch_size]
            inputs = extractor(
                batch,
                sampling_rate=HUBERT_SR,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            hidden = out.last_hidden_state  # (B, T, H)
            mask = inputs.get("attention_mask")
            if mask is not None:
                # Feature extractor mask is at waveform resolution; HuBERT
                # downsamples. Approximate by non-padding hidden frames via
                # lengths from attention_mask // hop (approx 320).
                lengths = mask.sum(dim=1)
                # Wav2Vec2/HuBERT CNN stride product ≈ 320 samples.
                feat_lens = torch.clamp(lengths // 320, min=1)
                pooled = []
                for b, fl in enumerate(feat_lens.tolist()):
                    fl = min(int(fl), hidden.shape[1])
                    pooled.append(hidden[b, :fl].mean(dim=0))
                pooled_t = torch.stack(pooled, dim=0)
            else:
                pooled_t = hidden.mean(dim=1)
            vectors.append(pooled_t.cpu().numpy().astype(np.float64))
            print(
                f"[hubert] {min(i + batch_size, len(clips_16k))}/{len(clips_16k)}",
                flush=True,
            )
    return np.vstack(vectors)


def pairwise_same_diff(
    embeddings: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (same_label_dists, different_label_dists) as 1D cosine distances."""
    from sklearn.metrics.pairwise import cosine_distances

    n = len(labels)
    if n < 2:
        return np.array([]), np.array([])
    D = cosine_distances(embeddings)
    same: list[float] = []
    diff: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(D[i, j])
            if labels[i] == labels[j]:
                same.append(d)
            else:
                diff.append(d)
    return np.asarray(same, dtype=np.float64), np.asarray(diff, dtype=np.float64)


def maybe_subsample(diff: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(diff) <= max_n:
        return diff
    idx = rng.choice(len(diff), size=max_n, replace=False)
    return diff[idx]


HIST_XLIM = (0.0, 2.0)
HIST_BINS = np.linspace(HIST_XLIM[0], HIST_XLIM[1], 41)


def plot_histograms(
    mel_same: np.ndarray,
    mel_diff: np.ndarray,
    hub_same: np.ndarray,
    hub_diff: np.ndarray,
    out_path: Path,
    *,
    title: str = "Same-label vs different-label pairwise distances",
    mel_title: str = "Mel fingerprint",
    hub_title: str = "HuBERT (mean-pool)",
) -> None:
    """1×2 mel | HuBERT histograms with shared xlim and ylim (density)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    panels = [
        (axes[0], mel_same, mel_diff, mel_title),
        (axes[1], hub_same, hub_diff, hub_title),
    ]
    for ax, same, diff, panel_title in panels:
        if len(same):
            ax.hist(
                same,
                bins=HIST_BINS,
                alpha=0.65,
                density=True,
                label=f"same-label (n={len(same)})",
                color="#2a6f97",
            )
        if len(diff):
            ax.hist(
                diff,
                bins=HIST_BINS,
                alpha=0.55,
                density=True,
                label=f"different-label (n={len(diff)})",
                color="#c1121f",
            )
        ax.set_title(panel_title)
        ax.set_xlabel("cosine distance (1 − cosine similarity)")
        ax.legend(fontsize=8)
        ax.set_xlim(*HIST_XLIM)
    axes[0].set_ylabel("density")
    # Lock shared ylim to the taller panel so density is comparable.
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0.0, ymax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _umap_xy(embeddings: np.ndarray) -> np.ndarray | None:
    try:
        import umap
    except ImportError:
        print(
            "[umap] umap-learn not installed — skip scatter. "
            "Install: tools/.venv/bin/pip install umap-learn",
            flush=True,
        )
        return None
    if len(embeddings) < 3:
        return None
    reducer = umap.UMAP(
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _scatter_umap_ax(ax, xy: np.ndarray, labels: list[str], *, show_legend: bool) -> None:
    import matplotlib.pyplot as plt

    uniq = sorted(set(labels))
    counts = Counter(labels)
    top = {lab for lab, _ in counts.most_common(20)}
    base = plt.colormaps["tab20"]
    color_of = {lab: base(i % 20) for i, lab in enumerate(sorted(top))}

    for lab in uniq:
        mask = np.array([l == lab for l in labels])
        if lab not in top:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=18, c="#bbbbbb", alpha=0.5, linewidths=0)
    for lab in sorted(top):
        mask = np.array([l == lab for l in labels])
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=28,
            c=[color_of[lab]],
            label=f"{lab} ({counts[lab]})" if show_legend else None,
            alpha=0.85,
            linewidths=0,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if show_legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.9, markerscale=1.2)


def _shared_xy_limits(*xys: np.ndarray, pad: float = 0.05) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = np.concatenate([xy[:, 0] for xy in xys])
    ys = np.concatenate([xy[:, 1] for xy in xys])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    return (xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy)


def plot_umap(
    embeddings: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> bool:
    """Return True if plot written, False if umap unavailable / too few points."""
    import matplotlib.pyplot as plt

    xy = _umap_xy(embeddings)
    if xy is None:
        if len(embeddings) < 3:
            print(f"[umap] too few points ({len(embeddings)}) for {title}; skip")
        return False

    fig, ax = plt.subplots(figsize=(8, 6.5))
    _scatter_umap_ax(ax, xy, labels, show_legend=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def plot_umap_side_by_side(
    mel_emb: np.ndarray,
    hub_emb: np.ndarray,
    labels: list[str],
    out_path: Path,
    *,
    title: str,
    mel_title: str = "Mel (norm on)",
    hub_title: str = "HuBERT (mean-pool)",
) -> bool:
    """1×2 mel | HuBERT UMAP with shared xlim/ylim (union of both fits)."""
    import matplotlib.pyplot as plt

    if len(mel_emb) < 3 or len(hub_emb) < 3:
        print(f"[umap] too few points for combined {title}; skip")
        return False
    mel_xy = _umap_xy(mel_emb)
    hub_xy = _umap_xy(hub_emb)
    if mel_xy is None or hub_xy is None:
        return False

    xlim, ylim = _shared_xy_limits(mel_xy, hub_xy)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    for ax, xy, panel_title, legend in (
        (axes[0], mel_xy, mel_title, True),
        (axes[1], hub_xy, hub_title, False),
    ):
        _scatter_umap_ax(ax, xy, labels, show_legend=legend)
        ax.set_title(panel_title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def mean_or_nan(xs: np.ndarray) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def summarize_space(name: str, same: np.ndarray, diff: np.ndarray) -> dict:
    ms, md = mean_or_nan(same), mean_or_nan(diff)
    gap = md - ms if (len(same) and len(diff)) else float("nan")
    return {
        "space": name,
        "mean_same": ms,
        "mean_diff": md,
        "gap": gap,
        "n_same": int(len(same)),
        "n_diff": int(len(diff)),
    }


def print_summary_rows(rows: list[dict], *, header: str = "summary") -> None:
    print(f"\n=== {header} (cosine distance) ===")
    print(
        f"{'space':<28} {'mean_same':>10} {'mean_diff':>10} "
        f"{'gap(diff-same)':>14} {'n_same':>8} {'n_diff':>8}"
    )
    for row in rows:
        print(
            f"{row['space']:<28} {row['mean_same']:10.4f} {row['mean_diff']:10.4f} "
            f"{row['gap']:14.4f} {row['n_same']:8d} {row['n_diff']:8d}"
        )
    print(
        "\nLarger gap (different − same) ⇒ better same-label clustering in that space."
    )


def load_clips(
    kit_hits: list[tuple[str, Path]],
) -> tuple[list[np.ndarray], list[int], list[str], list[dict], int]:
    import soundfile as sf

    clips: list[np.ndarray] = []
    clip_srs: list[int] = []
    labels: list[str] = []
    meta: list[dict] = []
    skipped_short = 0

    for needle, kit in kit_hits:
        tags = load_hand_tags(kit)
        audio_path = resolve_audio(kit)
        if not audio_path.exists():
            raise SystemExit(f"[error] missing audio {audio_path}")
        try:
            audio, sr = sf.read(str(audio_path), always_2d=False)
        except Exception as e:
            raise SystemExit(f"[error] failed to read {audio_path}: {e}") from e
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float64)
        sr = int(sr)
        print(f"[load] {kit.name}: {len(tags)} usable tags, sr={sr}", flush=True)

        for t in tags:
            clip = slice_samples(audio, sr, t["_start"], t["_end"])
            if clip is None:
                skipped_short += 1
                continue
            clips.append(clip)
            clip_srs.append(sr)
            labels.append(t["_label"])
            meta.append(
                {
                    "kit": kit.name,
                    "needle": needle,
                    "startMs": t["_start"],
                    "endMs": t["_end"],
                    "label": t["_label"],
                    "category": t["_category"],
                    "nonverbal": t["_nonverbal"],
                    "uuid": t.get("uuid"),
                }
            )
    return clips, clip_srs, labels, meta, skipped_short


def subset_mask(meta: list[dict], *, exclude_nonverbal: bool) -> np.ndarray:
    if not exclude_nonverbal:
        return np.ones(len(meta), dtype=bool)
    return np.array([not m["nonverbal"] for m in meta], dtype=bool)


def condition_slug(*, mel_norm: bool, exclude_nonverbal: bool) -> str:
    norm = "norm_on" if mel_norm else "norm_off"
    labels = "excl_nonverbal" if exclude_nonverbal else "all"
    return f"mel_{norm}_{labels}"


def run_condition_stats(
    embeddings: np.ndarray,
    labels: list[str],
    *,
    space_name: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    same, diff = pairwise_same_diff(embeddings, labels)
    return summarize_space(space_name, same, diff), same, diff


def _embed_cache_path(out_dir: Path) -> Path:
    return out_dir / "embeddings_cache.npz"


def _meta_fingerprint(meta: list[dict]) -> str:
    """Stable id so cache is invalidated if tag set / spans change."""
    import hashlib

    parts = [
        f"{m.get('kit')}|{m.get('uuid')}|{m['startMs']}|{m['endMs']}|{m['label']}|{int(m['nonverbal'])}"
        for m in meta
    ]
    blob = "\n".join(parts).encode("utf-8")
    return f"{len(parts)}:{hashlib.sha1(blob).hexdigest()}"


def load_or_compute_embeddings(
    *,
    clips: list[np.ndarray],
    clip_srs: list[int],
    meta: list[dict],
    out_dir: Path,
    hubert_batch: int,
    skip_hubert: bool,
    refresh_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mel_off, mel_on, hub). Reuses ``embeddings_cache.npz`` when valid."""
    n_all = len(clips)
    cache_path = _embed_cache_path(out_dir)
    fingerprint = _meta_fingerprint(meta)

    if not refresh_cache and cache_path.exists():
        try:
            z = np.load(cache_path, allow_pickle=False)
            if (
                str(z["fingerprint"]) == fingerprint
                and z["mel_off"].shape[0] == n_all
                and z["mel_on"].shape[0] == n_all
                and (skip_hubert or z["hub"].shape[0] == n_all)
            ):
                print(f"[cache] loaded embeddings from {cache_path}", flush=True)
                hub = z["hub"] if not skip_hubert else np.zeros((n_all, 2))
                return z["mel_off"], z["mel_on"], hub
            print("[cache] stale or mismatched — recomputing embeddings", flush=True)
        except Exception as e:
            print(f"[cache] load failed ({e}); recomputing", flush=True)

    print("[mel] embedding norm=OFF (production log_mel_embed)…", flush=True)
    mel_off = np.stack(
        [log_mel_embed_analysis(c, sr, normalize=False) for c, sr in zip(clips, clip_srs)],
        axis=0,
    )
    print("[mel] embedding norm=ON (RMS + no duration feature)…", flush=True)
    mel_on = np.stack(
        [log_mel_embed_analysis(c, sr, normalize=True) for c, sr in zip(clips, clip_srs)],
        axis=0,
    )

    if skip_hubert:
        hub = np.zeros((n_all, 2))
        print("[hubert] skipped")
    else:
        clips_16k = [resample_to(c, sr, HUBERT_SR) for c, sr in zip(clips, clip_srs)]
        hub = hubert_embed_batch(clips_16k, batch_size=hubert_batch)

    np.savez_compressed(
        cache_path,
        mel_off=mel_off,
        mel_on=mel_on,
        hub=hub,
        fingerprint=np.asarray(fingerprint),
    )
    print(f"[cache] wrote {cache_path}", flush=True)
    return mel_off, mel_on, hub


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / norms).astype(np.float64)


def fit_supervised_projector(
    X: np.ndarray,
    labels: list[str],
    *,
    max_components: int = LDA_MAX_COMPONENTS,
    seed: int = 42,
    train_protocol: str | None = None,
) -> tuple[object, dict]:
    """Fit a supervised projector; return ``(transform_fn, meta)``.

    ``transform_fn(X)`` returns L2-normalized rows. Prefer sklearn
    ``LinearDiscriminantAnalysis`` with
    ``n_components = min(n_classes-1, n_features, max_components)``.
    On failure, try PCA→LDA, then ``NeighborhoodComponentsAnalysis``.
    """
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(labels)
    n_classes = int(len(set(y.tolist())))
    n_samples, n_features = X.shape
    if n_classes < 2:
        raise RuntimeError(
            f"Need ≥2 train classes for LDA; got n_classes={n_classes}, "
            f"n_samples={n_samples}"
        )
    n_comp = max(1, min(n_classes - 1, n_features, max_components))
    counts = Counter(labels)
    n_singleton = sum(1 for c in counts.values() if c == 1)

    meta: dict = {
        "train_protocol": train_protocol
        or (
            "fit-on-all labeled excl_nonverbal clips "
            "(optimistic diagnostic upper bound; not leave-one-kit-out)"
        ),
        "n_samples": n_samples,
        "n_features_in": n_features,
        "n_classes": n_classes,
        "n_singleton_labels": n_singleton,
        "n_components_requested": n_comp,
        "l2_normalize_after": True,
        "max_components_cap": max_components,
    }

    errors: list[str] = []

    # --- Attempt 1: LDA (svd) ---
    try:
        lda = LinearDiscriminantAnalysis(n_components=n_comp, solver="svd")
        lda.fit(X, y)
        meta.update(
            {
                "method": "LDA",
                "solver": "svd",
                "n_components_out": int(lda.n_components),
                "fallback_chain": ["LDA"],
            }
        )

        def transform(X_new: np.ndarray, _lda=lda) -> np.ndarray:
            return _l2_normalize_rows(_lda.transform(X_new))

        # Confirm output dims from a tiny transform.
        meta["n_components_out"] = int(transform(X[:1]).shape[1])
        return transform, meta
    except Exception as e:
        errors.append(f"LDA(svd): {e}")
        print(f"[lda] LDA(svd) failed: {e}", flush=True)

    # --- Attempt 2: PCA → LDA (shrinkage / eigen more stable after PCA) ---
    try:
        pca_dim = min(n_samples - 1, n_features, max(n_comp * 4, 64))
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_pca = pca.fit_transform(X)
        n_comp2 = max(1, min(n_classes - 1, X_pca.shape[1], max_components))
        lda = LinearDiscriminantAnalysis(
            n_components=n_comp2, solver="eigen", shrinkage="auto"
        )
        lda.fit(X_pca, y)
        meta.update(
            {
                "method": "PCA+LDA",
                "solver": "eigen+shrinkage",
                "pca_dim": pca_dim,
                "n_components_out": int(lda.n_components),
                "fallback_chain": ["LDA", "PCA+LDA"],
                "prior_errors": errors,
            }
        )

        def transform(X_new: np.ndarray, _pca=pca, _lda=lda) -> np.ndarray:
            return _l2_normalize_rows(_lda.transform(_pca.transform(X_new)))

        meta["n_components_out"] = int(transform(X[:1]).shape[1])
        return transform, meta
    except Exception as e:
        errors.append(f"PCA+LDA: {e}")
        print(f"[lda] PCA+LDA failed: {e}", flush=True)

    # --- Attempt 3: NCA (small n_components) ---
    try:
        n_comp3 = max(1, min(n_comp, 16, n_features))
        nca = NeighborhoodComponentsAnalysis(
            n_components=n_comp3,
            random_state=seed,
            max_iter=100,
        )
        nca.fit(X, y)
        meta.update(
            {
                "method": "NCA",
                "n_components_out": int(n_comp3),
                "fallback_chain": ["LDA", "PCA+LDA", "NCA"],
                "prior_errors": errors,
            }
        )

        def transform(X_new: np.ndarray, _nca=nca) -> np.ndarray:
            return _l2_normalize_rows(_nca.transform(X_new))

        meta["n_components_out"] = int(transform(X[:1]).shape[1])
        return transform, meta
    except Exception as e:
        errors.append(f"NCA: {e}")
        raise RuntimeError(
            "All supervised projections failed:\n  " + "\n  ".join(errors)
        ) from e


def fit_supervised_projection(
    X: np.ndarray,
    labels: list[str],
    *,
    max_components: int = LDA_MAX_COMPONENTS,
    seed: int = 42,
    train_protocol: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit + transform all rows. Returns (X_proj L2-normalized, meta)."""
    transform, meta = fit_supervised_projector(
        X,
        labels,
        max_components=max_components,
        seed=seed,
        train_protocol=train_protocol,
    )
    return transform(X), meta


def _kit_slug(needle: str) -> str:
    """Filesystem-safe short id from a kit needle."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in needle)[:48]


def _stats_row(
    emb: np.ndarray, labs: list[str], *, space: str, n_clips: int | None = None
) -> dict:
    row, _, _ = run_condition_stats(emb, labs, space_name=space)
    if n_clips is not None:
        row["n_clips"] = n_clips
    return row


def run_lda(
    *,
    clips: list[np.ndarray],
    clip_srs: list[int],
    labels: list[str],
    meta: list[dict],
    out_dir: Path,
    hubert_batch: int,
    max_diff_pairs: int,
    seed: int,
    skip_umap: bool,
    skip_hubert: bool = True,
    refresh_cache: bool = False,
) -> int:
    """Compare raw mel (norm on) vs LDA-projected mel on excl_nonverbal."""
    rng = np.random.default_rng(seed)
    n_all = len(clips)
    n_nv = sum(1 for m in meta if m["nonverbal"])
    mask = subset_mask(meta, exclude_nonverbal=True)
    idx = np.where(mask)[0]
    labs = [labels[i] for i in idx]
    n_clips = int(mask.sum())
    print(
        f"[filter] clips before={n_all}  nonverbal_category={n_nv}  "
        f"after_exclude={n_clips}",
        flush=True,
    )
    if n_clips < 2:
        print("Need at least 2 excl_nonverbal clips.")
        return 1

    counts = Counter(labs)
    singletons = sorted(lab for lab, c in counts.items() if c == 1)
    multi = sorted(
        ((lab, c) for lab, c in counts.items() if c >= 2), key=lambda x: -x[1]
    )
    print(
        f"[labels] unique={len(counts)}  multi-instance={len(multi)}  "
        f"singletons={len(singletons)}",
        flush=True,
    )

    # Reuse matrix cache (mel_on preferred); skip HuBERT by default for LDA run.
    mel_off, mel_on, _hub = load_or_compute_embeddings(
        clips=clips,
        clip_srs=clip_srs,
        meta=meta,
        out_dir=out_dir,
        hubert_batch=hubert_batch,
        skip_hubert=True,
        refresh_cache=refresh_cache,
    )
    del mel_off  # LDA uses mel norm-on only
    mel_raw = mel_on[idx]
    print(
        f"[lda] base mel: norm_on, shape={mel_raw.shape}  "
        f"(excl_nonverbal, n={n_clips})",
        flush=True,
    )

    mel_lda, proj_meta = fit_supervised_projection(
        mel_raw, labs, max_components=LDA_MAX_COMPONENTS, seed=seed
    )
    print(
        f"[lda] method={proj_meta['method']}  "
        f"dims_in={proj_meta['n_features_in']} → "
        f"dims_out={proj_meta['n_components_out']}  "
        f"n_classes={proj_meta['n_classes']}  "
        f"protocol={proj_meta['train_protocol']}",
        flush=True,
    )

    raw_row, mel_same, mel_diff = run_condition_stats(
        mel_raw, labs, space_name="mel_norm_on_excl_nonverbal"
    )
    lda_row, lda_same, lda_diff = run_condition_stats(
        mel_lda, labs, space_name="mel_lda_excl_nonverbal"
    )
    raw_row["mel_norm"] = True
    raw_row["exclude_nonverbal"] = True
    raw_row["n_clips"] = n_clips
    lda_row["mel_norm"] = True
    lda_row["exclude_nonverbal"] = True
    lda_row["n_clips"] = n_clips
    lda_row["projection"] = proj_meta

    ordered = [raw_row, lda_row]
    print_summary_rows(ordered, header="LDA vs raw mel (excl_nonverbal)")
    gap_delta = (
        lda_row["gap"] - raw_row["gap"]
        if math.isfinite(lda_row["gap"]) and math.isfinite(raw_row["gap"])
        else float("nan")
    )
    widened = bool(math.isfinite(gap_delta) and gap_delta > 0)
    print(
        f"\n=== gap delta (LDA − raw mel) ===\n"
        f"  raw_gap={raw_row['gap']:.4f}  lda_gap={lda_row['gap']:.4f}  "
        f"Δ={gap_delta:+.4f}  widened={widened}",
        flush=True,
    )
    print(
        "\nNote: fit-on-all is an optimistic upper bound "
        "(labels used both to train the projection and to score gaps).",
        flush=True,
    )

    mel_diff_hist = maybe_subsample(mel_diff, max_diff_pairs, rng)
    lda_diff_hist = maybe_subsample(lda_diff, max_diff_pairs, rng)

    hist_path = out_dir / "hist_mel_vs_lda_excl_nonverbal.png"
    plot_histograms(
        mel_same,
        mel_diff_hist,
        lda_same,
        lda_diff_hist,
        hist_path,
        title=(
            f"Distances — raw mel (norm on) vs LDA mel, "
            f"excl_nonverbal (n={n_clips})"
        ),
        mel_title="Mel (norm on, raw)",
        hub_title=(
            f"Mel LDA ({proj_meta['method']}, "
            f"d={proj_meta['n_components_out']})"
        ),
    )
    print(f"[wrote] {hist_path}")

    plot_paths: dict = {"hist_mel_vs_lda_excl_nonverbal": str(hist_path)}
    umap_ok = True
    lda_umap_path = out_dir / "umap_lda_excl_nonverbal.png"
    combo_umap = out_dir / "umap_mel_vs_lda_excl_nonverbal.png"

    if skip_umap:
        print("[umap] skipped by flag")
        umap_ok = False
    else:
        if plot_umap(
            mel_lda,
            labs,
            lda_umap_path,
            (
                f"UMAP — mel LDA ({proj_meta['method']}, "
                f"d={proj_meta['n_components_out']}, excl_nonverbal, n={n_clips})"
            ),
        ):
            print(f"[wrote] {lda_umap_path}")
            plot_paths["umap_lda_excl_nonverbal"] = str(lda_umap_path)
        else:
            umap_ok = False

        if plot_umap_side_by_side(
            mel_raw,
            mel_lda,
            labs,
            combo_umap,
            title=(
                f"UMAP — raw mel (norm on) vs LDA mel "
                f"(excl_nonverbal, n={n_clips})"
            ),
            mel_title="Mel (norm on, raw)",
            hub_title=(
                f"Mel LDA ({proj_meta['method']}, "
                f"d={proj_meta['n_components_out']})"
            ),
        ):
            print(f"[wrote] {combo_umap}")
            plot_paths["umap_mel_vs_lda_excl_nonverbal"] = str(combo_umap)
        else:
            umap_ok = False

    summary_path = out_dir / "summary_lda.json"
    conclusion = (
        f"LDA ({proj_meta['method']}, d={proj_meta['n_components_out']}) "
        f"{'widened' if widened else 'did not widen'} the cosine gap "
        f"({raw_row['gap']:.4f} → {lda_row['gap']:.4f}, Δ={gap_delta:+.4f}); "
        f"fit-on-all is optimistic."
    )
    payload = {
        "mode": "lda",
        "n_clips_all": n_all,
        "n_clips_nonverbal": n_nv,
        "n_clips_excl_nonverbal": n_clips,
        "filter": {
            "exclude_nonverbal": "category == 'non-verbal vocalization' (case-insensitive)",
        },
        "mel_base": (
            f"analysis-only norm ON: RMS→{TARGET_RMS}, fixed {N_FRAMES_OUT}-frame "
            "mel grid, per-bin z-score, no duration feature, L2-normalize"
        ),
        "projection": proj_meta,
        "distance": "sklearn cosine_distances (= 1 − cosine similarity)",
        "label_key": "word → label → phonetic (lower/strip; skip empty/untitled)",
        "n_unique_labels": len(counts),
        "n_singleton_labels": len(singletons),
        "n_multi_labels": len(multi),
        "table": ordered,
        "gap_delta_lda_minus_raw": gap_delta,
        "gap_widened": widened,
        "conclusion": conclusion,
        "plots": plot_paths,
        "note": (
            "Fit-on-all labeled excl_nonverbal clips is an optimistic diagnostic "
            "upper bound (same labels train the projection and score the gap)."
        ),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[wrote] {summary_path}")
    print(f"\nCONCLUSION: {conclusion}")

    if not umap_ok and not skip_umap:
        return 2
    return 0


def run_lda_loko(
    *,
    clips: list[np.ndarray],
    clip_srs: list[int],
    labels: list[str],
    meta: list[dict],
    out_dir: Path,
    hubert_batch: int,
    max_diff_pairs: int,
    seed: int,
    skip_umap: bool,
    skip_hubert: bool = True,
    refresh_cache: bool = False,
) -> int:
    """Leave-one-kit-out (and optional held-out-label) LDA generalization check."""
    rng = np.random.default_rng(seed)
    n_all = len(clips)
    n_nv = sum(1 for m in meta if m["nonverbal"])
    mask = subset_mask(meta, exclude_nonverbal=True)
    idx = np.where(mask)[0]
    labs = [labels[i] for i in idx]
    meta_u = [meta[i] for i in idx]
    n_clips = int(mask.sum())
    print(
        f"[filter] clips before={n_all}  nonverbal_category={n_nv}  "
        f"after_exclude={n_clips}",
        flush=True,
    )
    if n_clips < 2:
        print("Need at least 2 excl_nonverbal clips.")
        return 1

    mel_off, mel_on, _hub = load_or_compute_embeddings(
        clips=clips,
        clip_srs=clip_srs,
        meta=meta,
        out_dir=out_dir,
        hubert_batch=hubert_batch,
        skip_hubert=True,
        refresh_cache=refresh_cache,
    )
    del mel_off
    mel_raw = mel_on[idx]
    needles = sorted({m["needle"] for m in meta_u})
    if len(needles) < 2:
        print(
            f"[error] leave-one-kit-out needs ≥2 kits; found {needles!r}",
            flush=True,
        )
        return 1

    # --- Optimistic fit-on-all reference (same as --lda) ---
    mel_lda_all, fit_all_meta = fit_supervised_projection(
        mel_raw,
        labs,
        max_components=LDA_MAX_COMPONENTS,
        seed=seed,
        train_protocol=(
            "fit-on-all labeled excl_nonverbal clips "
            "(optimistic diagnostic upper bound; not leave-one-kit-out)"
        ),
    )
    raw_overall = _stats_row(
        mel_raw, labs, space="mel_norm_on_excl_nonverbal", n_clips=n_clips
    )
    lda_fit_all = _stats_row(
        mel_lda_all, labs, space="mel_lda_fit_on_all", n_clips=n_clips
    )
    print(
        f"[lda-loko] fit-on-all reference: method={fit_all_meta['method']}  "
        f"d={fit_all_meta['n_components_out']}  "
        f"raw_gap={raw_overall['gap']:.4f}  "
        f"lda_gap={lda_fit_all['gap']:.4f}",
        flush=True,
    )

    # Unseen-label policy: project all held-out clips; primary gap uses all
    # pairs on H. Also report gap after dropping clips whose labels never
    # appeared in the train kit(s) (same-label stats then only involve labels
    # LDA was trained on).
    unseen_policy = (
        "Project all held-out clips (sklearn LDA.transform needs no labels). "
        "Primary gap: all pairs on kit H. Secondary gap: drop clips whose "
        "label was unseen in train (exclude those labels from same/diff stats)."
    )

    fold_rows: list[dict] = []
    table_rows: list[dict] = [
        {**raw_overall, "fold": "overall"},
        {**lda_fit_all, "fold": "overall", "projection": fit_all_meta},
    ]
    plot_paths: dict = {}
    umap_ok = True
    umap_done = False

    print("\n=== leave-one-kit-out folds ===", flush=True)
    for holdout in needles:
        te_mask = np.array([m["needle"] == holdout for m in meta_u], dtype=bool)
        tr_mask = ~te_mask
        n_te = int(te_mask.sum())
        n_tr = int(tr_mask.sum())
        labs_tr = [labs[i] for i in range(n_clips) if tr_mask[i]]
        labs_te = [labs[i] for i in range(n_clips) if te_mask[i]]
        X_tr = mel_raw[tr_mask]
        X_te = mel_raw[te_mask]
        train_labels = set(labs_tr)
        test_labels = set(labs_te)
        unseen_labels = sorted(test_labels - train_labels)
        seen_labels = sorted(test_labels & train_labels)
        unseen_clip_mask = np.array(
            [lab not in train_labels for lab in labs_te], dtype=bool
        )
        n_unseen_clips = int(unseen_clip_mask.sum())
        n_seen_clips = n_te - n_unseen_clips

        print(
            f"\n[fold] holdout={holdout!r}  train_n={n_tr}  test_n={n_te}  "
            f"train_classes={len(train_labels)}  test_labels={len(test_labels)}  "
            f"overlap_labels={len(seen_labels)}  unseen_labels={len(unseen_labels)}  "
            f"unseen_clips={n_unseen_clips}/{n_te}",
            flush=True,
        )

        transform, proj_meta = fit_supervised_projector(
            X_tr,
            labs_tr,
            max_components=LDA_MAX_COMPONENTS,
            seed=seed,
            train_protocol=(
                f"leave-one-kit-out: train on kits≠{holdout}, "
                f"score gap on held-out kit only"
            ),
        )
        X_te_lda = transform(X_te)

        raw_te = _stats_row(
            X_te, labs_te, space=f"mel_raw_holdout_{holdout}", n_clips=n_te
        )
        lda_te = _stats_row(
            X_te_lda, labs_te, space=f"mel_lda_loko_holdout_{holdout}", n_clips=n_te
        )

        # Secondary: only clips with labels seen in train
        seen_idx = np.where(~unseen_clip_mask)[0]
        if len(seen_idx) >= 2:
            raw_seen = _stats_row(
                X_te[seen_idx],
                [labs_te[i] for i in seen_idx],
                space=f"mel_raw_holdout_{holdout}_seen_labels",
                n_clips=n_seen_clips,
            )
            lda_seen = _stats_row(
                X_te_lda[seen_idx],
                [labs_te[i] for i in seen_idx],
                space=f"mel_lda_loko_holdout_{holdout}_seen_labels",
                n_clips=n_seen_clips,
            )
        else:
            raw_seen = {
                "space": f"mel_raw_holdout_{holdout}_seen_labels",
                "mean_same": float("nan"),
                "mean_diff": float("nan"),
                "gap": float("nan"),
                "n_same": 0,
                "n_diff": 0,
                "n_clips": n_seen_clips,
            }
            lda_seen = {
                "space": f"mel_lda_loko_holdout_{holdout}_seen_labels",
                "mean_same": float("nan"),
                "mean_diff": float("nan"),
                "gap": float("nan"),
                "n_same": 0,
                "n_diff": 0,
                "n_clips": n_seen_clips,
            }

        delta_vs_raw = (
            lda_te["gap"] - raw_te["gap"]
            if math.isfinite(lda_te["gap"]) and math.isfinite(raw_te["gap"])
            else float("nan")
        )
        delta_vs_fit_all = (
            lda_te["gap"] - lda_fit_all["gap"]
            if math.isfinite(lda_te["gap"]) and math.isfinite(lda_fit_all["gap"])
            else float("nan")
        )

        fold = {
            "holdout_kit": holdout,
            "n_train": n_tr,
            "n_test": n_te,
            "n_train_classes": len(train_labels),
            "n_test_labels": len(test_labels),
            "n_overlap_labels": len(seen_labels),
            "n_unseen_labels": len(unseen_labels),
            "n_unseen_clips": n_unseen_clips,
            "n_seen_clips": n_seen_clips,
            "unseen_labels": unseen_labels,
            "overlap_labels": seen_labels,
            "projection": proj_meta,
            "raw_holdout": raw_te,
            "lda_holdout": lda_te,
            "raw_holdout_seen_labels": raw_seen,
            "lda_holdout_seen_labels": lda_seen,
            "gap_delta_lda_minus_raw": delta_vs_raw,
            "gap_delta_lda_minus_fit_on_all": delta_vs_fit_all,
        }
        fold_rows.append(fold)
        table_rows.extend(
            [
                {**raw_te, "fold": holdout, "subset": "all_holdout"},
                {
                    **lda_te,
                    "fold": holdout,
                    "subset": "all_holdout",
                    "projection": proj_meta,
                },
                {**raw_seen, "fold": holdout, "subset": "seen_labels_only"},
                {
                    **lda_seen,
                    "fold": holdout,
                    "subset": "seen_labels_only",
                },
            ]
        )

        print(
            f"  raw_gap={raw_te['gap']:.4f}  lda_loko_gap={lda_te['gap']:.4f}  "
            f"Δvs_raw={delta_vs_raw:+.4f}  Δvs_fit_all={delta_vs_fit_all:+.4f}",
            flush=True,
        )
        print(
            f"  seen-label-only: n={n_seen_clips}  "
            f"raw_gap={raw_seen['gap']:.4f}  lda_gap={lda_seen['gap']:.4f}",
            flush=True,
        )

        # Optional UMAP for first fold only (cheap): holdout kit in LDA space.
        if not skip_umap and not umap_done and n_te >= 3:
            slug = _kit_slug(holdout)
            umap_path = out_dir / f"umap_lda_loko_holdout_{slug}.png"
            title = (
                f"UMAP — LDA LOKO holdout {holdout} "
                f"(train n={n_tr}, test n={n_te}, "
                f"d={proj_meta['n_components_out']})"
            )
            if plot_umap(X_te_lda, labs_te, umap_path, title):
                print(f"[wrote] {umap_path}")
                plot_paths[f"umap_lda_loko_holdout_{slug}"] = str(umap_path)
                umap_done = True
            else:
                umap_ok = False

    # Aggregate mean across folds (primary all-holdout LDA gaps)
    loko_gaps = [f["lda_holdout"]["gap"] for f in fold_rows]
    raw_fold_gaps = [f["raw_holdout"]["gap"] for f in fold_rows]
    loko_seen_gaps = [f["lda_holdout_seen_labels"]["gap"] for f in fold_rows]
    mean_loko = float(np.nanmean(loko_gaps)) if loko_gaps else float("nan")
    mean_raw_fold = float(np.nanmean(raw_fold_gaps)) if raw_fold_gaps else float("nan")
    mean_loko_seen = (
        float(np.nanmean(loko_seen_gaps)) if loko_seen_gaps else float("nan")
    )

    # --- Optional: held-out labels (~20% of multi-instance types) ---
    counts = Counter(labs)
    multi_labels = sorted(lab for lab, c in counts.items() if c >= 2)
    n_hold_labels = max(1, int(round(0.20 * len(multi_labels)))) if multi_labels else 0
    held_out_label_result: dict | None = None
    if n_hold_labels >= 1 and len(multi_labels) >= 2:
        shuffled = list(multi_labels)
        rng.shuffle(shuffled)
        held_labels = set(shuffled[:n_hold_labels])
        train_lab_mask = np.array([lab not in held_labels for lab in labs], dtype=bool)
        test_lab_mask = ~train_lab_mask
        n_tr_l = int(train_lab_mask.sum())
        n_te_l = int(test_lab_mask.sum())
        print(
            f"\n=== held-out labels (seed={seed}) ===\n"
            f"  multi-instance labels={len(multi_labels)}  "
            f"held_out={n_hold_labels} ({sorted(held_labels)})  "
            f"train_clips={n_tr_l}  held_clips={n_te_l}",
            flush=True,
        )
        if n_tr_l >= 2 and n_te_l >= 2 and len(set(labs[i] for i in range(n_clips) if train_lab_mask[i])) >= 2:
            labs_tr_l = [labs[i] for i in range(n_clips) if train_lab_mask[i]]
            labs_te_l = [labs[i] for i in range(n_clips) if test_lab_mask[i]]
            transform_l, proj_l = fit_supervised_projector(
                mel_raw[train_lab_mask],
                labs_tr_l,
                max_components=LDA_MAX_COMPONENTS,
                seed=seed,
                train_protocol=(
                    f"held-out labels: train on {len(multi_labels) - n_hold_labels}/"
                    f"{len(multi_labels)} multi-instance label types "
                    f"(+ all singleton train labels); seed={seed}"
                ),
            )
            X_te_l = transform_l(mel_raw[test_lab_mask])
            # Also transform all for overall gap after projection trained w/o held labels
            X_all_l = transform_l(mel_raw)
            raw_held = _stats_row(
                mel_raw[test_lab_mask],
                labs_te_l,
                space="mel_raw_held_out_labels",
                n_clips=n_te_l,
            )
            lda_held = _stats_row(
                X_te_l,
                labs_te_l,
                space="mel_lda_held_out_labels",
                n_clips=n_te_l,
            )
            raw_all_after = _stats_row(
                mel_raw, labs, space="mel_raw_overall", n_clips=n_clips
            )
            lda_all_after = _stats_row(
                X_all_l, labs, space="mel_lda_trained_wo_held_labels_overall", n_clips=n_clips
            )
            held_out_label_result = {
                "seed": seed,
                "n_multi_labels": len(multi_labels),
                "n_held_out_labels": n_hold_labels,
                "held_out_labels": sorted(held_labels),
                "n_train_clips": n_tr_l,
                "n_held_clips": n_te_l,
                "projection": proj_l,
                "raw_held_out_label_clips": raw_held,
                "lda_held_out_label_clips": lda_held,
                "raw_overall": raw_all_after,
                "lda_overall_trained_wo_held": lda_all_after,
                "gap_delta_held_lda_minus_raw": (
                    lda_held["gap"] - raw_held["gap"]
                    if math.isfinite(lda_held["gap"]) and math.isfinite(raw_held["gap"])
                    else float("nan")
                ),
            }
            table_rows.extend(
                [
                    {**raw_held, "fold": "held_out_labels", "subset": "held_labels"},
                    {
                        **lda_held,
                        "fold": "held_out_labels",
                        "subset": "held_labels",
                        "projection": proj_l,
                    },
                ]
            )
            print(
                f"  held-label clips: raw_gap={raw_held['gap']:.4f}  "
                f"lda_gap={lda_held['gap']:.4f}  "
                f"Δ={held_out_label_result['gap_delta_held_lda_minus_raw']:+.4f}",
                flush=True,
            )
            print(
                f"  overall after train-wo-held: raw_gap={raw_all_after['gap']:.4f}  "
                f"lda_gap={lda_all_after['gap']:.4f}",
                flush=True,
            )
        else:
            print("[held-out labels] skipped (too few train/test clips)", flush=True)
    else:
        print("[held-out labels] skipped (need ≥2 multi-instance labels)", flush=True)

    # Print comparison table
    print("\n=== LOKO summary table (cosine distance) ===")
    print(
        f"{'space':<42} {'mean_same':>10} {'mean_diff':>10} "
        f"{'gap':>10} {'n_same':>8} {'n_diff':>8}"
    )
    summary_print = [
        ("raw_mel_overall", raw_overall),
        ("lda_fit_on_all", lda_fit_all),
    ]
    for f in fold_rows:
        h = f["holdout_kit"]
        summary_print.append((f"raw_holdout[{h}]", f["raw_holdout"]))
        summary_print.append((f"lda_loko[{h}]", f["lda_holdout"]))
        summary_print.append(
            (f"lda_loko_seen[{h}]", f["lda_holdout_seen_labels"])
        )
    summary_print.append(
        (
            "mean_raw_holdout_folds",
            {
                "mean_same": float("nan"),
                "mean_diff": float("nan"),
                "gap": mean_raw_fold,
                "n_same": 0,
                "n_diff": 0,
            },
        )
    )
    summary_print.append(
        (
            "mean_lda_loko_folds",
            {
                "mean_same": float("nan"),
                "mean_diff": float("nan"),
                "gap": mean_loko,
                "n_same": 0,
                "n_diff": 0,
            },
        )
    )
    summary_print.append(
        (
            "mean_lda_loko_seen_folds",
            {
                "mean_same": float("nan"),
                "mean_diff": float("nan"),
                "gap": mean_loko_seen,
                "n_same": 0,
                "n_diff": 0,
            },
        )
    )
    if held_out_label_result is not None:
        summary_print.append(
            ("raw_held_out_labels", held_out_label_result["raw_held_out_label_clips"])
        )
        summary_print.append(
            ("lda_held_out_labels", held_out_label_result["lda_held_out_label_clips"])
        )

    for name, row in summary_print:
        ms, md, g = row["mean_same"], row["mean_diff"], row["gap"]
        ms_s = f"{ms:10.4f}" if math.isfinite(ms) else f"{'nan':>10}"
        md_s = f"{md:10.4f}" if math.isfinite(md) else f"{'nan':>10}"
        g_s = f"{g:10.4f}" if math.isfinite(g) else f"{'nan':>10}"
        print(
            f"{name:<42} {ms_s} {md_s} {g_s} "
            f"{row['n_same']:8d} {row['n_diff']:8d}"
        )

    print(
        f"\nReferences: raw overall gap≈{raw_overall['gap']:.4f}  "
        f"fit-on-all LDA gap≈{lda_fit_all['gap']:.4f}",
        flush=True,
    )
    print(f"Unseen-label policy: {unseen_policy}", flush=True)

    # Conclusion: LOKO should beat raw; fit-on-all is optimistic upper bound.
    # Held-out-label gap is the stricter test of whether class directions
    # transfer to novel words (what clustering would need for new tags).
    beats_raw = bool(math.isfinite(mean_loko) and mean_loko > mean_raw_fold)
    held_gap = (
        held_out_label_result["lda_held_out_label_clips"]["gap"]
        if held_out_label_result is not None
        else float("nan")
    )
    held_raw = (
        held_out_label_result["raw_held_out_label_clips"]["gap"]
        if held_out_label_result is not None
        else float("nan")
    )
    held_helps = (
        math.isfinite(held_gap)
        and math.isfinite(held_raw)
        and held_gap >= held_raw + 0.05
    )
    strong_loko = bool(
        math.isfinite(mean_loko)
        and math.isfinite(lda_fit_all["gap"])
        and mean_loko >= 0.75 * lda_fit_all["gap"]
        and mean_loko >= raw_overall["gap"] + 0.08
    )
    if beats_raw and strong_loko and held_helps:
        verdict = "generalizes well enough to consider for clustering"
        consider = True
    elif beats_raw and mean_loko >= raw_overall["gap"] + 0.05:
        verdict = (
            "partial cross-kit gain over raw, but not enough for clustering "
            "(LOKO << fit-on-all and/or held-out labels barely improve; "
            f"cross-kit label overlap is tiny — "
            f"{fold_rows[0]['n_overlap_labels']} shared labels)"
        )
        consider = False
    elif beats_raw:
        verdict = (
            "weak generalization (LOKO barely above raw holdout; "
            "not enough to consider for clustering)"
        )
        consider = False
    else:
        verdict = (
            "does not generalize (LOKO gap ≤ raw holdout); "
            "do not use LDA for clustering"
        )
        consider = False

    conclusion = (
        f"LOKO mean LDA gap={mean_loko:.4f} vs mean raw holdout={mean_raw_fold:.4f} "
        f"vs fit-on-all={lda_fit_all['gap']:.4f} (overall raw={raw_overall['gap']:.4f})"
        + (
            f"; held-out labels gap={held_gap:.4f} vs raw={held_raw:.4f}"
            if math.isfinite(held_gap)
            else ""
        )
        + f"; {verdict}."
    )

    summary_path = out_dir / "summary_lda_loko.json"
    payload = {
        "mode": "lda_loko",
        "n_clips_all": n_all,
        "n_clips_nonverbal": n_nv,
        "n_clips_excl_nonverbal": n_clips,
        "kits": needles,
        "filter": {
            "exclude_nonverbal": "category == 'non-verbal vocalization' (case-insensitive)",
        },
        "mel_base": (
            f"analysis-only norm ON: RMS→{TARGET_RMS}, fixed {N_FRAMES_OUT}-frame "
            "mel grid, per-bin z-score, no duration feature, L2-normalize"
        ),
        "distance": "sklearn cosine_distances (= 1 − cosine similarity)",
        "label_key": "word → label → phonetic (lower/strip; skip empty/untitled)",
        "unseen_label_policy": unseen_policy,
        "references": {
            "raw_mel_overall": raw_overall,
            "lda_fit_on_all": lda_fit_all,
            "fit_on_all_projection": fit_all_meta,
        },
        "folds": fold_rows,
        "aggregate": {
            "mean_raw_holdout_gap": mean_raw_fold,
            "mean_lda_loko_gap": mean_loko,
            "mean_lda_loko_seen_labels_gap": mean_loko_seen,
            "per_fold_lda_gaps": loko_gaps,
            "per_fold_raw_gaps": raw_fold_gaps,
        },
        "held_out_labels": held_out_label_result,
        "table": table_rows,
        "consider_for_clustering": consider,
        "conclusion": conclusion,
        "plots": plot_paths,
        "seed": seed,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[wrote] {summary_path}")
    print(f"\nCONCLUSION: {conclusion}")
    print(f"consider_for_clustering={consider}")

    if not umap_ok and not skip_umap:
        return 2
    return 0


def run_matrix(
    *,
    clips: list[np.ndarray],
    clip_srs: list[int],
    labels: list[str],
    meta: list[dict],
    out_dir: Path,
    hubert_batch: int,
    max_diff_pairs: int,
    seed: int,
    skip_umap: bool,
    skip_hubert: bool,
    refresh_cache: bool = False,
) -> int:
    rng = np.random.default_rng(seed)
    n_all = len(clips)
    n_nv = sum(1 for m in meta if m["nonverbal"])
    n_keep = n_all - n_nv
    print(
        f"[filter] clips before={n_all}  nonverbal_category={n_nv}  "
        f"after_exclude={n_keep}",
        flush=True,
    )

    mel_off, mel_on, hub = load_or_compute_embeddings(
        clips=clips,
        clip_srs=clip_srs,
        meta=meta,
        out_dir=out_dir,
        hubert_batch=hubert_batch,
        skip_hubert=skip_hubert,
        refresh_cache=refresh_cache,
    )

    label_sets = (
        ("all", False),
        ("excl_nonverbal", True),
    )
    mel_modes = (
        ("norm_off", False, mel_off),
        ("norm_on", True, mel_on),
    )

    summary_rows: list[dict] = []
    conditions: dict = {}
    umap_ok = True
    plot_paths: dict = {}

    for label_slug, excl in label_sets:
        mask = subset_mask(meta, exclude_nonverbal=excl)
        idx = np.where(mask)[0]
        labs = [labels[i] for i in idx]
        hub_sub = hub[idx]
        mel_on_sub = mel_on[idx]
        n_clips = int(mask.sum())

        if skip_hubert:
            hub_row = summarize_space(f"hubert_{label_slug}", np.array([]), np.array([]))
            hub_same, hub_diff = np.array([]), np.array([])
        else:
            hub_row, hub_same, hub_diff = run_condition_stats(
                hub_sub, labs, space_name=f"hubert_{label_slug}"
            )
        summary_rows.append(hub_row)
        hub_diff_hist = maybe_subsample(hub_diff, max_diff_pairs, rng)

        hub_umap_path = out_dir / f"umap_hubert_{label_slug}.png"
        if not skip_umap and not skip_hubert:
            if plot_umap(
                hub_sub,
                labs,
                hub_umap_path,
                f"UMAP — HuBERT ({label_slug}, n={n_clips})",
            ):
                print(f"[wrote] {hub_umap_path}")
                plot_paths[f"umap_hubert_{label_slug}"] = str(hub_umap_path)
            else:
                umap_ok = False

        # Combined shared-scale figures: mel norm-on | HuBERT
        if not skip_hubert:
            mel_on_same, mel_on_diff = pairwise_same_diff(mel_on_sub, labs)
            mel_on_diff_hist = maybe_subsample(mel_on_diff, max_diff_pairs, rng)
            combo_hist = out_dir / f"hist_mel_vs_hubert_{label_slug}.png"
            plot_histograms(
                mel_on_same,
                mel_on_diff_hist,
                hub_same,
                hub_diff_hist,
                combo_hist,
                title=(
                    f"Distances — mel (norm on) vs HuBERT, "
                    f"labels={label_slug} (n={n_clips})"
                ),
                mel_title="Mel (norm on)",
                hub_title="HuBERT (mean-pool)",
            )
            print(f"[wrote] {combo_hist}")
            plot_paths[f"hist_mel_vs_hubert_{label_slug}"] = str(combo_hist)

            if not skip_umap:
                combo_umap = out_dir / f"umap_mel_vs_hubert_{label_slug}.png"
                if plot_umap_side_by_side(
                    mel_on_sub,
                    hub_sub,
                    labs,
                    combo_umap,
                    title=f"UMAP — mel (norm on) vs HuBERT ({label_slug}, n={n_clips})",
                    mel_title="Mel (norm on)",
                    hub_title="HuBERT (mean-pool)",
                ):
                    print(f"[wrote] {combo_umap}")
                    plot_paths[f"umap_mel_vs_hubert_{label_slug}"] = str(combo_umap)
                else:
                    umap_ok = False

        for norm_slug, mel_norm, mel_full in mel_modes:
            mel_sub = mel_full[idx]
            space = f"mel_{norm_slug}_{label_slug}"
            mel_row, mel_same, mel_diff = run_condition_stats(
                mel_sub, labs, space_name=space
            )
            mel_row["mel_norm"] = mel_norm
            mel_row["exclude_nonverbal"] = excl
            mel_row["n_clips"] = n_clips
            summary_rows.append(mel_row)

            mel_diff_hist = maybe_subsample(mel_diff, max_diff_pairs, rng)
            hist_name = f"hist_mel_{norm_slug}_{label_slug}.png"
            hist_path = out_dir / hist_name
            plot_histograms(
                mel_same,
                mel_diff_hist,
                hub_same,
                hub_diff_hist,
                hist_path,
                title=(
                    f"Distances — mel {norm_slug.replace('_', ' ')}, "
                    f"labels={label_slug} (n={n_clips})"
                ),
                mel_title=f"Mel ({norm_slug.replace('_', ' ')})",
            )
            print(f"[wrote] {hist_path}")
            plot_paths[hist_name] = str(hist_path)

            mel_umap_path = out_dir / f"umap_mel_{norm_slug}_{label_slug}.png"
            if not skip_umap:
                if plot_umap(
                    mel_sub,
                    labs,
                    mel_umap_path,
                    f"UMAP — mel {norm_slug} ({label_slug}, n={n_clips})",
                ):
                    print(f"[wrote] {mel_umap_path}")
                    plot_paths[f"umap_mel_{norm_slug}_{label_slug}"] = str(mel_umap_path)
                else:
                    umap_ok = False

            conditions[space] = {
                **mel_row,
                "hubert": hub_row,
                "plots": {
                    "histogram": str(hist_path),
                    "hist_mel_vs_hubert": str(
                        out_dir / f"hist_mel_vs_hubert_{label_slug}.png"
                    ),
                    "umap_mel": str(mel_umap_path) if mel_umap_path.exists() else None,
                    "umap_hubert": (
                        str(hub_umap_path) if hub_umap_path.exists() else None
                    ),
                    "umap_mel_vs_hubert": str(
                        out_dir / f"umap_mel_vs_hubert_{label_slug}.png"
                    ),
                },
            }

    # Order table: mel matrix then hubert
    mel_rows = [r for r in summary_rows if r["space"].startswith("mel_")]
    hub_rows = [r for r in summary_rows if r["space"].startswith("hubert_")]
    # Desired mel order: off/all, on/all, off/excl, on/excl
    order = [
        "mel_norm_off_all",
        "mel_norm_on_all",
        "mel_norm_off_excl_nonverbal",
        "mel_norm_on_excl_nonverbal",
        "hubert_all",
        "hubert_excl_nonverbal",
    ]
    by_name = {r["space"]: r for r in mel_rows + hub_rows}
    ordered = [by_name[k] for k in order if k in by_name]
    print_summary_rows(ordered, header="matrix summary")

    # Gap deltas vs production baseline (mel_norm_off_all)
    base = by_name.get("mel_norm_off_all")
    if base and math.isfinite(base["gap"]):
        print("\n=== mel gap deltas vs mel_norm_off_all ===")
        for key in (
            "mel_norm_on_all",
            "mel_norm_off_excl_nonverbal",
            "mel_norm_on_excl_nonverbal",
        ):
            row = by_name.get(key)
            if not row or not math.isfinite(row["gap"]):
                continue
            delta = row["gap"] - base["gap"]
            print(f"  {key:<32} gap={row['gap']:.4f}  Δ={delta:+.4f}")

    if "hubert_all" in by_name and "hubert_excl_nonverbal" in by_name:
        ha, hx = by_name["hubert_all"], by_name["hubert_excl_nonverbal"]
        if math.isfinite(ha["gap"]) and math.isfinite(hx["gap"]):
            print(
                f"\n=== hubert gap: all={ha['gap']:.4f}  "
                f"excl_nonverbal={hx['gap']:.4f}  "
                f"Δ={hx['gap'] - ha['gap']:+.4f}"
            )

    summary_path = out_dir / "summary.json"
    payload = {
        "mode": "matrix",
        "n_clips_all": n_all,
        "n_clips_nonverbal": n_nv,
        "n_clips_excl_nonverbal": n_keep,
        "filter": {
            "exclude_nonverbal": "category == 'non-verbal vocalization' (case-insensitive)",
            "dropped_from": "embed set (distances, histograms, UMAP)",
        },
        "mel_normalization": {
            "off": "production log_mel_embed (peak norm + fixed 32-frame grid + log10(duration_ms))",
            "on": (
                f"analysis-only: RMS→{TARGET_RMS}, fixed {N_FRAMES_OUT}-frame mel grid, "
                "per-bin z-score, no duration feature, L2-normalize"
            ),
            "duration_choice": (
                "resample mel spectrogram to fixed n_frames_out via linear "
                "interpolation (same as production), then omit duration scalar"
            ),
        },
        "distance": "sklearn cosine_distances (= 1 − cosine similarity)",
        "label_key": "word → label → phonetic (lower/strip; skip empty/untitled)",
        "shared_scale_plots": {
            "hist_mel_vs_hubert_*": (
                "1×2 mel(norm on)|HuBERT; density=True; shared xlim=[0,2]; "
                "shared ylim=max density across panels"
            ),
            "umap_mel_vs_hubert_*": (
                "1×2 mel(norm on)|HuBERT; shared xlim/ylim = padded union of both UMAP fits"
            ),
        },
        "table": ordered,
        "conditions": conditions,
        "plots": plot_paths,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[wrote] {summary_path}")

    if not umap_ok and not skip_umap:
        return 2
    return 0


def run_single(
    *,
    clips: list[np.ndarray],
    clip_srs: list[int],
    labels: list[str],
    meta: list[dict],
    out_dir: Path,
    mel_norm: bool,
    exclude_nonverbal: bool,
    hubert_batch: int,
    max_diff_pairs: int,
    seed: int,
    skip_umap: bool,
    skip_hubert: bool,
) -> int:
    rng = np.random.default_rng(seed)
    mask = subset_mask(meta, exclude_nonverbal=exclude_nonverbal)
    n_before = len(clips)
    n_nv = sum(1 for m in meta if m["nonverbal"])
    idx = np.where(mask)[0]
    clips_u = [clips[i] for i in idx]
    srs_u = [clip_srs[i] for i in idx]
    labels_u = [labels[i] for i in idx]
    n = len(clips_u)
    print(
        f"[filter] before={n_before}  nonverbal={n_nv}  after={n}  "
        f"exclude_nonverbal={exclude_nonverbal}",
        flush=True,
    )
    if n < 2:
        print("Need at least 2 clips to compare distances.")
        return 1

    counts = Counter(labels_u)
    singletons = sorted(lab for lab, c in counts.items() if c == 1)
    multi = sorted(((lab, c) for lab, c in counts.items() if c >= 2), key=lambda x: -x[1])
    print(
        f"[labels] unique={len(counts)}  multi-instance={len(multi)}  "
        f"singletons={len(singletons)}"
    )

    print(f"[mel] embedding normalize={mel_norm}…", flush=True)
    mel = np.stack(
        [
            log_mel_embed_analysis(c, sr, normalize=mel_norm)
            for c, sr in zip(clips_u, srs_u)
        ],
        axis=0,
    )
    mel_same, mel_diff = pairwise_same_diff(mel, labels_u)
    mel_diff_hist = maybe_subsample(mel_diff, max_diff_pairs, rng)

    if skip_hubert:
        hub = np.zeros((n, 2))
        hub_same, hub_diff = np.array([]), np.array([])
        hub_diff_hist = hub_diff
        print("[hubert] skipped")
    else:
        clips_16k = [resample_to(c, sr, HUBERT_SR) for c, sr in zip(clips_u, srs_u)]
        hub = hubert_embed_batch(clips_16k, batch_size=hubert_batch)
        hub_same, hub_diff = pairwise_same_diff(hub, labels_u)
        hub_diff_hist = maybe_subsample(hub_diff, max_diff_pairs, rng)

    rows = [
        summarize_space("mel", mel_same, mel_diff),
        summarize_space("hubert", hub_same, hub_diff),
    ]
    print_summary_rows(rows)

    slug = condition_slug(mel_norm=mel_norm, exclude_nonverbal=exclude_nonverbal)
    hist_path = out_dir / f"hist_{slug}.png"
    plot_histograms(mel_same, mel_diff_hist, hub_same, hub_diff_hist, hist_path)
    print(f"[wrote] {hist_path}")

    umap_ok = True
    mel_umap = out_dir / f"umap_mel_{slug}.png"
    hub_umap = out_dir / f"umap_hubert_{slug}.png"
    if skip_umap:
        print("[umap] skipped by flag")
        umap_ok = False
    else:
        if plot_umap(mel, labels_u, mel_umap, f"UMAP — mel ({slug})"):
            print(f"[wrote] {mel_umap}")
        else:
            umap_ok = False
        if not skip_hubert:
            if plot_umap(hub, labels_u, hub_umap, f"UMAP — HuBERT ({slug})"):
                print(f"[wrote] {hub_umap}")
            else:
                umap_ok = False

    summary = {r["space"]: r for r in rows}
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "mode": "single",
                "mel_norm": mel_norm,
                "exclude_nonverbal": exclude_nonverbal,
                "n_clips_before": n_before,
                "n_clips_nonverbal": n_nv,
                "n_clips": n,
                "skipped_short": 0,
                "label_key": "word → label → phonetic (lower/strip; skip empty/untitled)",
                "distance": "sklearn cosine_distances (= 1 − cosine similarity)",
                "n_unique_labels": len(counts),
                "n_singleton_labels": len(singletons),
                "n_multi_labels": len(multi),
                "summary": summary,
                "plots": {
                    "histograms": str(hist_path),
                    "umap_mel": str(mel_umap) if mel_umap.exists() else None,
                    "umap_hubert": str(hub_umap) if hub_umap.exists() else None,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[wrote] {summary_path}")
    if not umap_ok and not skip_umap:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(LIBRARY_DIR))
    ap.add_argument(
        "--needle",
        action="append",
        dest="needles",
        help="Kit match string (repeatable). Default: the two curated sessions.",
    )
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--hubert-batch", type=int, default=8)
    ap.add_argument("--max-diff-pairs", type=int, default=MAX_DIFF_PAIRS_HIST)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-umap", action="store_true")
    ap.add_argument("--skip-hubert", action="store_true", help="Mel-only (debug)")
    ap.add_argument(
        "--mel-norm",
        action="store_true",
        help="Analysis-only RMS loudness + drop duration feature for mel path",
    )
    ap.add_argument(
        "--exclude-nonverbal",
        action="store_true",
        help="Drop tags with category 'non-verbal vocalization' before embedding",
    )
    ap.add_argument(
        "--matrix",
        action="store_true",
        help=(
            "Run full condition matrix: mel norm on/off × all/excl-nonverbal, "
            "plus HuBERT with/without nonverbal"
        ),
    )
    ap.add_argument(
        "--lda",
        action="store_true",
        help=(
            "Supervised LDA (or PCA+LDA / NCA fallback) on mel norm-on "
            "excl_nonverbal; compare cosine gap + UMAP vs raw mel"
        ),
    )
    ap.add_argument(
        "--lda-loko",
        action="store_true",
        help=(
            "Leave-one-kit-out LDA generalization on mel norm-on excl_nonverbal; "
            "also optional held-out-label split; writes summary_lda_loko.json"
        ),
    )
    ap.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore embeddings_cache.npz and recompute mel/HuBERT embeddings",
    )
    args = ap.parse_args(argv)

    mode_flags = sum(bool(x) for x in (args.matrix, args.lda, args.lda_loko))
    if mode_flags > 1:
        raise SystemExit("Use only one of --matrix, --lda, or --lda-loko.")

    # LDA modes are mel-only by default; avoid requiring torch/transformers.
    skip_hubert = True if (args.lda or args.lda_loko) else args.skip_hubert
    _require_deps(need_hubert=not skip_hubert, need_umap=not args.skip_umap)

    library = Path(args.library).expanduser()
    needles = args.needles or list(DEFAULT_NEEDLES)
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = (
            OUT_DIR_MATRIX
            if (args.matrix or args.lda or args.lda_loko)
            else OUT_DIR
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    kit_hits = find_kits(library, needles)
    print("[kits]")
    for needle, kit in kit_hits:
        print(f"  {needle!r} → {kit.name}")

    clips, clip_srs, labels, meta, skipped_short = load_clips(kit_hits)
    print(
        f"[clips] n={len(clips)}  skipped_short={skipped_short}",
        flush=True,
    )
    if len(clips) < 2:
        print("Need at least 2 clips to compare distances.")
        return 1

    common = dict(
        clips=clips,
        clip_srs=clip_srs,
        labels=labels,
        meta=meta,
        out_dir=out_dir,
        hubert_batch=args.hubert_batch,
        max_diff_pairs=args.max_diff_pairs,
        seed=args.seed,
        skip_umap=args.skip_umap,
        skip_hubert=skip_hubert,
    )
    if args.lda_loko:
        return run_lda_loko(**common, refresh_cache=args.refresh_cache)
    if args.lda:
        return run_lda(**common, refresh_cache=args.refresh_cache)
    if args.matrix:
        return run_matrix(**common, refresh_cache=args.refresh_cache)
    return run_single(
        **common,
        mel_norm=args.mel_norm,
        exclude_nonverbal=args.exclude_nonverbal,
    )


if __name__ == "__main__":
    raise SystemExit(main())
