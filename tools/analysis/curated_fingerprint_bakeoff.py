#!/usr/bin/env python3
"""Curated-cluster fingerprint bake-off with leave-one-kit-out (LOKO).

Compares frozen audio embeddings on the same curated multi-member clusters
used by ``curated_cluster_learn.py`` (SHORT dropped; nonverbal excluded).

Spaces (installable without heroic effort):
  * mel — production ``cluster_sounds.log_mel_embed`` (baseline)
  * hubert — facebook/hubert-base-ls960 (reference; known weaker)
  * yamnet — AudioSet YAMNet (torch-vggish-yamnet)
  * vggish — AudioSet VGGish (torchvggish)
  * panns — PANNs Cnn14 embedding (panns-inference)
  * byola — BYOL-A AudioNTT2020 512-d (vendored encoder + official weights)

Skipped (documented in report): Audio-JEPA / EAT — no pip-installable package.

For each space reports:
  * pooled within / between / gap (all kits)
  * leave-one-kit-out: gap on the holdout kit only; mean ± std across kits

No label-supervised training — encoders stay frozen fingerprints.

Usage
-----
  tools/.venv/bin/python tools/analysis/curated_fingerprint_bakeoff.py
  tools/.venv/bin/python tools/analysis/curated_fingerprint_bakeoff.py \\
      --library ~/Documents/BabyTalk/Library \\
      --out tools/analysis/out/fingerprint_bakeoff
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from babytalk_paths import LIBRARY_DIR, list_local_kits  # noqa: E402
from cluster_sounds import log_mel_embed  # noqa: E402
from curated_cluster_learn import (  # noqa: E402
    build_member_rows,
    discover_curated_clusters,
    gap_table_row,
    stats_1d,
)
from embed_compare_mel_hubert import (  # noqa: E402
    HUBERT_SR,
    hubert_embed_batch,
    pairwise_same_diff,
    resample_to,
    rms_normalize,
)

OUT_DIR = Path(__file__).resolve().parent / "out" / "fingerprint_bakeoff"
BYOLA_WEIGHT_DEFAULT = OUT_DIR / "byola" / "AudioNTT2020-BYOLA-64x96d512.pth"

# Cache downloads inside the out tree when possible.
os.environ.setdefault("TORCH_HOME", str(OUT_DIR / ".torch_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))


# ---------------------------------------------------------------------------
# BYOL-A (minimal vendored encoder — official AudioNTT2020 + weight load)
# ---------------------------------------------------------------------------


class _NetworkCommonMixIn:
    def load_weight(self, weight_file, device, state_dict=None, key_check=True):
        state_dict = state_dict or __import__("torch").load(
            weight_file, map_location=device, weights_only=False
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if key_check:
            weights = {}
            for k in state_dict:
                m = re.search(r"(^fc\.|\.fc\.|^features\.|\.features\.)", k)
                if m is None:
                    continue
                new_k = k[m.start() :]
                new_k = new_k[1:] if new_k[0] == "." else new_k
                weights[new_k] = state_dict[k]
        else:
            weights = state_dict
        self.load_state_dict(weights)
        self.eval()
        return self


class _AudioNTT2020Task6(__import__("torch").nn.Module, _NetworkCommonMixIn):
    def __init__(self, n_mels, d):
        import torch.nn as nn

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x):
        import torch

        x = self.features(x)
        x = x.permute(0, 3, 2, 1)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C * D))
        x = self.fc(x)
        return x


class AudioNTT2020(_AudioNTT2020Task6):
    def __init__(self, n_mels=64, d=512):
        super().__init__(n_mels=n_mels, d=d)

    def forward(self, x):
        import torch

        x = super().forward(x)
        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2
        return x


def _pad_min_seconds(samples: np.ndarray, sr: int, min_sec: float) -> np.ndarray:
    need = int(np.ceil(min_sec * sr))
    x = np.asarray(samples, dtype=np.float32)
    if len(x) >= need:
        return x
    return np.pad(x, (0, need - len(x)))


def _mean_vec(mat: np.ndarray) -> np.ndarray:
    """Mean-pool over leading batch/frame dims → 1D feature vector.

    For tensors shaped ``[N, D, 1, 1]`` (YAMNet), squeeze trailing singleton
    spatial dims *before* averaging so we keep the ``D`` feature axis.
    """
    v = np.asarray(mat, dtype=np.float64)
    # Squeeze trailing 1×1 spatial dims common in CNN embeddings.
    while v.ndim >= 3 and v.shape[-1] == 1 and v.shape[-2] == 1:
        v = v[..., 0, 0]
    if v.ndim == 1:
        return v.reshape(-1)
    if v.ndim == 2:
        # [N_frames, D] → mean over frames
        return v.mean(axis=0).reshape(-1)
    # Fallback: flatten trailing dims into features, mean over leading.
    lead = v.shape[0]
    flat = v.reshape(lead, -1)
    return flat.mean(axis=0).reshape(-1)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


def embed_mel(rows: list[dict]) -> np.ndarray:
    vecs = []
    for i, r in enumerate(rows):
        vecs.append(log_mel_embed(r["clip"], r["sr"]))
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[mel] {i + 1}/{len(rows)}", flush=True)
    return np.vstack(vecs)


def embed_hubert(rows: list[dict]) -> np.ndarray:
    clips = []
    for r in rows:
        x = rms_normalize(r["clip"])
        clips.append(resample_to(x, r["sr"], HUBERT_SR))
    print(f"[hubert] embedding {len(clips)} clips…", flush=True)
    return hubert_embed_batch(clips, batch_size=8, device="cpu")


def embed_yamnet(rows: list[dict]) -> np.ndarray:
    import torch
    from torch_vggish_yamnet.input_proc import WaveformToInput
    from torch_vggish_yamnet.yamnet.model import yamnet as yamnet_fn

    model = yamnet_fn(pretrained=True)
    model.eval()
    proc = WaveformToInput()
    vecs = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            wav = _pad_min_seconds(r["clip"], r["sr"], 0.98)
            wav_t = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
            patches = proc(wav_t, int(r["sr"]))
            if patches.shape[0] == 0:
                # Extremely short after resampling — pad harder.
                wav = _pad_min_seconds(r["clip"], r["sr"], 2.0)
                wav_t = torch.from_numpy(wav).float().unsqueeze(0)
                patches = proc(wav_t, int(r["sr"]))
            emb, _logits = model(patches)
            # emb: [N_patches, 1024, 1, 1]
            vecs.append(_mean_vec(emb.cpu().numpy()))
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"[yamnet] {i + 1}/{len(rows)}", flush=True)
    return np.vstack(vecs)


def embed_vggish(rows: list[dict]) -> np.ndarray:
    import torch
    from torchvggish import vggish, vggish_input

    model = vggish()
    model.eval()
    vecs = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            wav = _pad_min_seconds(r["clip"], r["sr"], 1.0)
            wav16 = resample_to(wav, r["sr"], 16000)
            examples = vggish_input.waveform_to_examples(wav16, 16000)
            if not isinstance(examples, torch.Tensor):
                examples = torch.from_numpy(np.asarray(examples)).float()
            else:
                examples = examples.float()
            if examples.ndim == 3:
                examples = examples.unsqueeze(1) if examples.shape[1] != 1 else examples
            # Ensure [N, 1, 96, 64]
            if examples.ndim == 4 and examples.shape[1] != 1 and examples.shape[-1] == 64:
                pass
            emb = model.forward(examples)
            vecs.append(_mean_vec(emb.cpu().numpy()))
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"[vggish] {i + 1}/{len(rows)}", flush=True)
    return np.vstack(vecs)


def embed_panns(rows: list[dict]) -> np.ndarray:
    from panns_inference import AudioTagging

    at = AudioTagging(checkpoint_path=None, device="cpu")
    vecs = []
    for i, r in enumerate(rows):
        wav = _pad_min_seconds(r["clip"], r["sr"], 0.5)
        wav32 = resample_to(wav, r["sr"], 32000)
        _clipwise, emb = at.inference(wav32[None, :])
        vecs.append(_mean_vec(np.asarray(emb)))
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[panns] {i + 1}/{len(rows)}", flush=True)
    return np.vstack(vecs)


def embed_byola(rows: list[dict], weight_path: Path) -> np.ndarray:
    """BYOL-A frozen encoder with per-clip log-mel normalization."""
    import torch
    import torchaudio.transforms as T

    if not weight_path.is_file():
        raise FileNotFoundError(
            f"BYOL-A weights missing: {weight_path}\n"
            "Download AudioNTT2020-BYOLA-64x96d512.pth from "
            "https://github.com/nttcslab/byol-a/tree/master/pretrained_weights"
        )

    # Match config.yaml: 16 kHz, n_fft=1024, hop=160, n_mels=64, shape [64, 96]
    to_mel = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        n_mels=64,
        f_min=60,
        f_max=7800,
    )
    model = AudioNTT2020(n_mels=64, d=512)
    model.load_weight(str(weight_path), device="cpu")
    model.eval()

    target_frames = 96
    vecs = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            wav = _pad_min_seconds(r["clip"], r["sr"], 0.95)
            wav16 = resample_to(wav, r["sr"], 16000)
            x = torch.from_numpy(wav16).float()
            mel = to_mel(x)  # [n_mels, T]
            lms = (mel + 1e-6).log()
            # Instance normalize (no dataset stats available)
            mu = lms.mean()
            std = lms.std().clamp_min(1e-6)
            lms = (lms - mu) / std
            Tlen = lms.shape[-1]
            if Tlen < target_frames:
                lms = torch.nn.functional.pad(lms, (0, target_frames - Tlen))
            elif Tlen > target_frames:
                # Center crop
                start = (Tlen - target_frames) // 2
                lms = lms[:, start : start + target_frames]
            inp = lms.unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
            emb = model(inp)
            vecs.append(emb.squeeze(0).cpu().numpy().astype(np.float64))
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"[byola] {i + 1}/{len(rows)}", flush=True)
    return np.vstack(vecs)


# ---------------------------------------------------------------------------
# Gap / LOKO
# ---------------------------------------------------------------------------


def gap_on_subset(
    embeddings: np.ndarray, labels: list[str], mask: np.ndarray
) -> dict | None:
    """Within/between/gap for rows where mask is True. None if not scorables."""
    idxs = np.where(mask)[0]
    if len(idxs) < 4:
        return None
    sub_emb = embeddings[idxs]
    sub_lab = [labels[i] for i in idxs]
    # Need ≥2 distinct clusters each with ≥2 members for a real gap.
    from collections import Counter

    counts = Counter(sub_lab)
    multi = [k for k, n in counts.items() if n >= 2]
    if len(multi) < 2:
        return None
    # Keep only members of multi-member clusters (same rule as pooled pairs
    # among those clusters).
    keep = [i for i, lab in enumerate(sub_lab) if counts[lab] >= 2]
    if len(keep) < 4:
        return None
    sub_emb2 = sub_emb[keep]
    sub_lab2 = [sub_lab[i] for i in keep]
    if len(set(sub_lab2)) < 2:
        return None
    same, diff = pairwise_same_diff(sub_emb2, sub_lab2)
    if len(same) == 0 or len(diff) == 0:
        return None
    return gap_table_row("subset", same, diff)


def leave_one_kit_out(
    embeddings: np.ndarray, rows: list[dict]
) -> dict:
    kits = sorted({r["kit"] for r in rows})
    labels = [r["cluster_key"] for r in rows]
    folds = []
    for holdout in kits:
        mask = np.asarray([r["kit"] == holdout for r in rows], dtype=bool)
        g = gap_on_subset(embeddings, labels, mask)
        if g is None:
            folds.append(
                {
                    "holdout_kit": holdout,
                    "n_members": int(mask.sum()),
                    "scorable": False,
                    "gap": None,
                    "mean_within": None,
                    "mean_between": None,
                    "n_within": 0,
                    "n_between": 0,
                    "reason": "need ≥2 multi-member curated clusters in holdout kit",
                }
            )
            continue
        folds.append(
            {
                "holdout_kit": holdout,
                "n_members": int(mask.sum()),
                "scorable": True,
                "gap": g["gap"],
                "mean_within": g["mean_within"],
                "mean_between": g["mean_between"],
                "n_within": g["n_within"],
                "n_between": g["n_between"],
            }
        )
    scored = [f for f in folds if f["scorable"] and f["gap"] is not None]
    gaps = np.asarray([f["gap"] for f in scored], dtype=np.float64)
    return {
        "n_kits": len(kits),
        "n_scorable_folds": len(scored),
        "mean_gap": float(np.mean(gaps)) if len(gaps) else None,
        "std_gap": float(np.std(gaps)) if len(gaps) else None,
        "folds": folds,
    }


# ---------------------------------------------------------------------------
# Report (beginner-friendly)
# ---------------------------------------------------------------------------


def _fmt(x, digits=3):
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def write_beginner_report(path: Path, summary: dict) -> None:
    disc = summary.get("discovery") or {}
    filt = summary.get("filters") or {}
    spaces = summary.get("spaces") or []
    skipped = summary.get("skipped") or []
    mewehv = summary.get("mewehv") or {}

    # Rank by LOKO mean gap (fallback pooled)
    ranked = sorted(
        [s for s in spaces if s.get("status") == "ok"],
        key=lambda s: (
            -(s.get("loko") or {}).get("mean_gap")
            if (s.get("loko") or {}).get("mean_gap") is not None
            else -1e9
        ),
    )

    lines: list[str] = []
    lines.append("# Fingerprint bake-off: which sound fingerprint best matches your curated words?")
    lines.append("")
    lines.append(
        "> Machine numbers: `summary.json` next to this file. "
        "Script: `tools/analysis/curated_fingerprint_bakeoff.py`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Why this report exists")
    lines.append("")
    lines.append(
        "BabyTalk’s Clustering tab groups clips using a **fingerprint** — "
        "a short list of numbers that summarize how a clip sounds. Production "
        "uses a **mel** fingerprint today. Other models (YAMNet, VGGish, PANNs, "
        "BYOL-A, HuBERT, …) make different fingerprints."
    )
    lines.append("")
    lines.append(
        "We already know mel separates your **human curated word groups** with "
        "a pooled gap of about **0.15**. This bake-off asks: *do any frozen "
        "off-the-shelf fingerprints do better — including when we leave one "
        "recording kit out?*"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## How to read the numbers (read this before the table)")
    lines.append("")
    lines.append("### Fingerprint / embedding")
    lines.append("")
    lines.append(
        "A **fingerprint** (also called an **embedding**) is a vector of numbers "
        "for one clip. Similar-sounding clips should get similar fingerprints. "
        "Here every model is **frozen** — we do **not** train or fine-tune on "
        "your labels. We only measure how well the ready-made fingerprint "
        "already separates human word groups."
    )
    lines.append("")
    lines.append("### Distance, within, between")
    lines.append("")
    lines.append(
        "**Distance** = how different two fingerprints are (cosine distance). "
        "Lower = more alike."
    )
    lines.append("")
    lines.append(
        "- **Within** — pairs in the *same* curated cluster (two “boot”s). We want this **small**."
    )
    lines.append(
        "- **Between** — pairs from *different* clusters (“boot” vs “fisch”). We want this **larger**."
    )
    lines.append("")
    lines.append("### Gap (the headline score)")
    lines.append("")
    lines.append("**Gap = average between − average within.**")
    lines.append("")
    lines.append(
        "Bigger gap → same-word clips clump together and different words sit farther apart → "
        "**better for clustering**. Tiny gap → the fingerprint doesn’t separate your human groups well."
    )
    lines.append("")
    lines.append(
        "Compare **gaps across models**, not raw within numbers — different models live on different scales."
    )
    lines.append("")
    lines.append("### Pooled vs leave-one-kit-out (LOKO)")
    lines.append("")
    lines.append(
        "- **Pooled** — put every clip from every kit in one big pile, then measure gap. "
        "This is the familiar ~0.15 mel number from the curated-clusters report. "
        "It can look a bit optimistic if kits differ a lot."
    )
    lines.append(
        "- **Leave-one-kit-out (LOKO)** — for each kit in turn, measure the gap **using only that kit’s clips**. "
        "Then average those kit gaps. Nothing is “trained” on the other kits (encoders are frozen); "
        "LOKO just answers: *does this fingerprint still separate words on a kit you didn’t mix into the pooled score?*"
    )
    lines.append("")
    lines.append(
        "When people say “compare to mel’s 0.15,” check **both** columns: pooled ≈ 0.15 for mel, "
        "and mel’s **LOKO mean** for a fair head-to-head with other fingerprints."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## What we looked at")
    lines.append("")
    lines.append(
        "Same curated filters as the earlier curated-clusters study:"
    )
    lines.append("")
    lines.append(
        f"- Kits kept: **{disc.get('n_kits_kept', '?')}** "
        f"({', '.join(disc.get('kits_kept') or []) or 'none'})"
    )
    lines.append(
        f"- Curated clusters (≥2 members after filters): **{disc.get('n_clusters_kept', '?')}**"
    )
    lines.append(
        f"- Clips kept: **{filt.get('kept', '?')}** "
        f"(dropped SHORT={filt.get('dropped_short', 0)}, "
        f"nonverbal={filt.get('dropped_nonverbal_tag', 0)})"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Results: pooled + LOKO gaps")
    lines.append("")
    lines.append(
        "| Fingerprint | Pooled within | Pooled between | **Pooled gap** | **LOKO mean gap** | LOKO std | Scorable kits |"
    )
    lines.append(
        "|-------------|--------------:|---------------:|---------------:|------------------:|---------:|--------------:|"
    )
    for s in spaces:
        if s.get("status") != "ok":
            lines.append(
                f"| {s['name']} | — | — | — | — | — | skipped ({s.get('skip_reason', '?')}) |"
            )
            continue
        p = s.get("pooled") or {}
        loko = s.get("loko") or {}
        lines.append(
            f"| {s['name']} | {_fmt(p.get('mean_within'))} | {_fmt(p.get('mean_between'))} | "
            f"**{_fmt(p.get('gap'))}** | **{_fmt(loko.get('mean_gap'))}** | "
            f"{_fmt(loko.get('std_gap'))} | {loko.get('n_scorable_folds', 0)}/{loko.get('n_kits', 0)} |"
        )
    lines.append("")

    mel = next((s for s in spaces if s["name"] == "mel" and s.get("status") == "ok"), None)
    if mel:
        lines.append(
            f"**Mel baseline:** pooled gap **{_fmt((mel.get('pooled') or {}).get('gap'))}**, "
            f"LOKO mean gap **{_fmt((mel.get('loko') or {}).get('mean_gap'))}** "
            f"(± {_fmt((mel.get('loko') or {}).get('std_gap'))} across "
            f"{(mel.get('loko') or {}).get('n_scorable_folds', 0)} kits)."
        )
        lines.append("")

    if ranked:
        best = ranked[0]
        lines.append("### Plain-language takeaway")
        lines.append("")
        best_loko = (best.get("loko") or {}).get("mean_gap")
        mel_loko = (mel.get("loko") or {}).get("mean_gap") if mel else None
        if mel and best["name"] == "mel":
            lines.append(
                "On this curated set, **mel still leads** (or ties) on leave-one-kit-out. "
                "Do **not** switch production clustering to another frozen fingerprint based on this bake-off alone."
            )
        elif mel and best_loko is not None and mel_loko is not None:
            delta = best_loko - mel_loko
            if delta > 0.01:
                lines.append(
                    f"**{best['name']}** has the highest LOKO mean gap "
                    f"({_fmt(best_loko)} vs mel {_fmt(mel_loko)}, Δ={_fmt(delta)}). "
                    "That is interesting — worth a careful re-listen / optional A/B in Clustering — "
                    "but keep mel as the default until you see the same win on more curated kits."
                )
            else:
                lines.append(
                    f"Best LOKO gap is **{best['name']}** ({_fmt(best_loko)}), "
                    f"essentially tied with mel ({_fmt(mel_loko)}). "
                    "Stay on mel for production."
                )
        lines.append("")

    # Per-kit LOKO for mel (so the user sees the protocol)
    if mel and (mel.get("loko") or {}).get("folds"):
        lines.append("### Mel LOKO by kit (so “0.15” isn’t confused)")
        lines.append("")
        lines.append("| Holdout kit | Members | Within | Between | Gap |")
        lines.append("|-------------|--------:|-------:|--------:|----:|")
        for f in mel["loko"]["folds"]:
            short = f["holdout_kit"]
            if len(short) > 40:
                short = short[:37] + "…"
            if not f.get("scorable"):
                lines.append(f"| {short} | {f.get('n_members', 0)} | — | — | unscorable |")
            else:
                lines.append(
                    f"| {short} | {f.get('n_members', 0)} | {_fmt(f.get('mean_within'))} | "
                    f"{_fmt(f.get('mean_between'))} | {_fmt(f.get('gap'))} |"
                )
        lines.append("")

    if skipped:
        lines.append("### Skipped fingerprints")
        lines.append("")
        for sk in skipped:
            lines.append(f"- **{sk['name']}** — {sk['reason']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## MeWEHV (paper check)")
    lines.append("")
    lines.append(
        mewehv.get(
            "plain",
            "MeWEHV combines a frozen wave encoder with an MFCC/CNN path for "
            "speaker / language / accent ID — not word clustering.",
        )
    )
    lines.append("")
    lines.append(f"**Recommendation:** {mewehv.get('recommendation', 'Skip for now.')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## FAQ — VAD parents/children, SSL, and what to do in Review")
    lines.append("")
    lines.append("### 5. VAD status — parents vs children (why Clustering looked like syllable mush)")
    lines.append("")
    lines.append(
        "BabyTalk’s current speech-finding pipeline is roughly:"
    )
    lines.append("")
    lines.append(
        "1. **Energy VAD + speechlike** — find louder-than-the-room regions that also sound like a voice "
        "(not taps/doors/water)."
    )
    lines.append(
        "2. **ECAPA (optional)** — cut those regions by speaker so one speaker’s talk isn’t mixed with another’s."
    )
    lines.append(
        "3. **Pause-split** — if a same-speaker span is still very long, cut on silences."
    )
    lines.append(
        "4. **DJW resegment** — de Jong & Wempe syllable-nucleus cuts turn a longer span into "
        "**syllable-ish children** that become ML candidates in Review."
    )
    lines.append("")
    lines.append(
        "**Parents** = the longer pre-resegment spans (VAD / speaker / pause pieces). "
        "**Children** = the shorter DJW pieces Review usually shows as “speech segments.”"
    )
    lines.append("")
    lines.append(
        "So when you open Review’s speech segments, you are mostly looking at **children** — "
        "often **syllables or short scraps**, not clean dictionary words. Clustering then "
        "fingerprints and groups those scraps. That is why auto-clusters can look like "
        "**syllable mush**: the input units are syllable-sized by design. Your **curated** "
        "clusters are different — you grouped word-like takes by hand — which is why this "
        "bake-off uses curated groups as ground truth, not the raw auto mush."
    )
    lines.append("")
    lines.append("### 7. Two SSL ideas in beginner terms")
    lines.append("")
    lines.append("**SSL** = self-supervised learning: a model learns useful sound patterns from lots of audio **without** word labels, then you reuse it.")
    lines.append("")
    lines.append(
        "1. **Frozen HuBERT (or YAMNet/…) fingerprint bake-off** — what this report did: "
        "download a pretrained model, freeze it, turn each clip into a fingerprint, measure gap. "
        "Cheap to try; no training on your library."
    )
    lines.append(
        "2. **Training / adapting on all library audio** — run SSL (or fine-tuning) on *your* "
        "recordings so the fingerprint learns BabyTalk’s rooms, mics, kids, and languages. "
        "More work; only worth it once you have enough clean curated word groups to prove it helped."
    )
    lines.append("")
    lines.append(
        "**“SSL across datasets”** would mean mixing BabyTalk kits with other public child/adult "
        "speech corpora during that adaptation step so the model generalizes better. "
        "**Wait** until you have more curated multi-member clusters: without a solid gap/LOKO "
        "test set, you can’t tell if the extra training helped word clustering or just shuffled numbers."
    )
    lines.append("")
    lines.append("### Review workflow — segments, clusters, or both?")
    lines.append("")
    lines.append(
        "- **Review ML speech segments (children)** when you want more *candidates* — accept / edit / dismiss "
        "syllable-ish proposals so tags exist. Expect fragments; merge or retag into real words as you listen."
    )
    lines.append(
        "- **Curate clusters** when you already have a few good tags of the same word — group them, name the cluster. "
        "That curated set is the ground truth we use for fingerprint experiments."
    )
    lines.append(
        "- **Both, in that order:** segments → clean tags → curated clusters. Don’t spend hours on Clustering "
        "auto-mush until you’ve curated a few solid word groups per kit."
    )
    lines.append("")
    lines.append("### When to look at new fingerprints?")
    lines.append("")
    lines.append(
        "After this bake-off: only chase a new fingerprint if its **LOKO gap clearly beats mel** "
        "on more kits, or if a product need appears (e.g. emotion / vocalization type — different job). "
        "Otherwise grow curated clusters and keep mel."
    )
    lines.append("")
    lines.append("### BYOL-A / YAMNet now vs after bake-off?")
    lines.append("")
    lines.append(
        "**After** — this report *is* that bake-off. Read the table above; don’t install them into "
        "production Clustering unless LOKO says they’re better. Trying them “just because” before "
        "numbers wastes Review time."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## What you should do this week")
    lines.append("")
    lines.append(
        "1. **Keep mel** as the Clustering default unless a row above clearly wins LOKO by a meaningful margin "
        "and still looks good when you re-listen."
    )
    lines.append(
        "2. **In Review:** confirm/dismiss speech-segment children to grow tags; then **curate** multi-member "
        "word clusters (the fuel for the next bake-off). Spot-check loose curated words from the earlier report "
        "(*brown*, *bus*, *fuchs*, …)."
    )
    lines.append(
        "3. **Skip MeWEHV** and skip SSL-across-datasets until you have more curated clusters; "
        "re-run this bake-off script when the curated set grows."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Generated from `curated_fingerprint_bakeoff.py`. "
        f"Spaces tried: {', '.join(s['name'] for s in spaces)}.*"
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_mewehv_block() -> dict:
    return {
        "paper": "https://arxiv.org/abs/2209.14078",
        "title": "MeWEHV: Mel and Wave Embeddings for Human Voice Tasks",
        "plain": (
            "**MeWEHV** (Mel and Wave Embeddings for Human Voice Tasks) is a research recipe that "
            "glues together two views of the same clip: (1) a big pretrained **wave** encoder "
            "(things like HuBERT / WavLM / Wav2Vec-style models) and (2) a small network that reads "
            "**MFCCs** (a classic speech feature related to mel). The combo is aimed at "
            "**who is speaking**, **which language**, and **which accent** — not at grouping "
            "toddler word takes like “boot” vs “fisch.”"
        ),
        "helps_word_clustering": False,
        "might_help_elsewhere": (
            "vocalization type / emotion / child-directed vs adult speech *only if* you later "
            "build labeled datasets for those jobs — still not the next lever for word Clustering."
        ),
        "recommendation": (
            "**Don’t use MeWEHV now for BabyTalk word clustering.** It solves a different problem "
            "(speaker/language/accent), adds training complexity, and we already have a simpler "
            "frozen-fingerprint bake-off path. Revisit only if you pivot to speaker/language ID."
        ),
        "one_liner": (
            "MeWEHV mixes wave+MFCC embeddings for speaker/language/accent ID — not word clustering; skip for now."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, default=LIBRARY_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--min-members", type=int, default=2)
    ap.add_argument("--include-nonverbal", action="store_true")
    ap.add_argument("--include-short", action="store_true")
    ap.add_argument("--skip-hubert", action="store_true")
    ap.add_argument("--skip-yamnet", action="store_true")
    ap.add_argument("--skip-vggish", action="store_true")
    ap.add_argument("--skip-panns", action="store_true")
    ap.add_argument("--skip-byola", action="store_true")
    ap.add_argument("--byola-weights", type=Path, default=BYOLA_WEIGHT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    library = args.library.expanduser()
    out_dir = args.out
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".mplconfig").mkdir(exist_ok=True)
    (out_dir / ".torch_cache").mkdir(exist_ok=True)
    os.environ["TORCH_HOME"] = str(out_dir / ".torch_cache")
    os.environ["MPLCONFIGDIR"] = str(out_dir / ".mplconfig")

    exclude_nv = not args.include_nonverbal
    exclude_short = not args.include_short

    print(f"[scan] library={library}", flush=True)
    curated = discover_curated_clusters(
        library, min_members=args.min_members, exclude_nonverbal=exclude_nv
    )
    kits_scanned = sum(
        1 for k in list_local_kits(library) if (k / "clusters.json").exists()
    )
    if not curated:
        raise SystemExit("No curated/labeled multi-member clusters found.")

    rows, filter_stats = build_member_rows(
        curated,
        exclude_short=exclude_short,
        exclude_nonverbal_members=exclude_nv,
    )
    print(
        f"[filter] kept {filter_stats['kept']} members in "
        f"{filter_stats['clusters_kept']} clusters",
        flush=True,
    )
    if filter_stats["clusters_kept"] < 2 or filter_stats["kept"] < 4:
        raise SystemExit("Too few members/clusters after filters.")

    cluster_keys = [r["cluster_key"] for r in rows]
    kits_kept = sorted({r["kit"] for r in rows})

    # Backend plan
    plan: list[tuple[str, callable | None, str | None]] = [
        ("mel", lambda: embed_mel(rows), None),
    ]
    if not args.skip_hubert:
        plan.append(("hubert", lambda: embed_hubert(rows), None))
    else:
        plan.append(("hubert", None, "skipped by flag"))
    if not args.skip_yamnet:
        plan.append(("yamnet", lambda: embed_yamnet(rows), None))
    else:
        plan.append(("yamnet", None, "skipped by flag"))
    if not args.skip_vggish:
        plan.append(("vggish", lambda: embed_vggish(rows), None))
    else:
        plan.append(("vggish", None, "skipped by flag"))
    if not args.skip_panns:
        plan.append(("panns", lambda: embed_panns(rows), None))
    else:
        plan.append(("panns", None, "skipped by flag"))
    if not args.skip_byola:
        plan.append(
            (
                "byola",
                lambda: embed_byola(rows, args.byola_weights.expanduser()),
                None,
            )
        )
    else:
        plan.append(("byola", None, "skipped by flag"))

    skipped = [
        {
            "name": "Audio-JEPA / EAT",
            "reason": (
                "No clearly installable pip package / weights path in tools/.venv "
                "without cloning research codebases; skipped (not heroic effort)."
            ),
        }
    ]

    spaces_out: list[dict] = []
    embeds_cache: dict[str, np.ndarray] = {}

    for name, fn, pre_skip in plan:
        if fn is None:
            spaces_out.append(
                {"name": name, "status": "skipped", "skip_reason": pre_skip}
            )
            continue
        try:
            print(f"[embed] starting {name}…", flush=True)
            mat = fn()
            embeds_cache[name] = mat
            same, diff = pairwise_same_diff(mat, cluster_keys)
            pooled = gap_table_row(name, same, diff)
            loko = leave_one_kit_out(mat, rows)
            spaces_out.append(
                {
                    "name": name,
                    "status": "ok",
                    "dim": int(mat.shape[1]),
                    "pooled": pooled,
                    "loko": loko,
                }
            )
            print(
                f"[gap] {name}: pooled={pooled['gap']:.4f} "
                f"loko_mean={loko['mean_gap']}",
                flush=True,
            )
        except Exception as e:
            print(f"[embed] {name} FAILED: {e}", flush=True)
            spaces_out.append(
                {
                    "name": name,
                    "status": "skipped",
                    "skip_reason": f"{type(e).__name__}: {e}",
                }
            )

    # Optional: cache embeddings for reruns
    cache_path = out_dir / "embeddings_cache.npz"
    if embeds_cache:
        np.savez_compressed(
            cache_path,
            **{k: v for k, v in embeds_cache.items()},
            cluster_key=np.asarray(cluster_keys),
            kit=np.asarray([r["kit"] for r in rows]),
        )

    discovery = {
        "n_kits_scanned": kits_scanned,
        "n_kits_with_curated_raw": len({c["kit"] for c in curated}),
        "n_kits_kept": len(kits_kept),
        "kits_kept": kits_kept,
        "n_clusters_raw": len(curated),
        "n_clusters_kept": filter_stats["clusters_kept"],
        "n_members_raw": filter_stats["members_raw"],
        "n_members_kept": filter_stats["kept"],
        "labels": sorted({c["label"] for c in curated}),
    }

    mewehv = write_mewehv_block()

    summary = {
        "mode": "curated_fingerprint_bakeoff_loko",
        "library": str(library),
        "discovery": discovery,
        "filters": filter_stats,
        "protocol": {
            "distance": "sklearn cosine_distances (= 1 − cosine similarity)",
            "same_key": "curated cluster id (kit::cluster_id)",
            "pooled": "all kits together",
            "loko": (
                "for each holdout kit, gap using only that kit’s members; "
                "require ≥2 multi-member curated clusters in the holdout kit; "
                "report mean/std of scorable fold gaps. Encoders frozen — no label training."
            ),
        },
        "spaces": spaces_out,
        "skipped": skipped,
        "mewehv": mewehv,
        "embeddings_cache": str(cache_path) if embeds_cache else None,
        "weekly_workflow": [
            "Keep mel as Clustering default unless a fingerprint clearly wins LOKO on more kits.",
            "Review: confirm speech-segment children → curate multi-member word clusters; spot-check loose words.",
            "Skip MeWEHV and SSL-across-datasets until curated set grows; re-run this bake-off then.",
        ],
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = out_dir / "report.md"
    write_beginner_report(report_path, summary)
    print(f"[done] wrote {summary_path}", flush=True)
    print(f"[done] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
