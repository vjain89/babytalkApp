#!/usr/bin/env python3
"""Reverse-engineer manually curated / labeled clusters via acoustic embeddings.

Unlike ``embed_compare_mel_hubert.py`` (same-``word`` string pairs on tags),
this study uses **human-grouped cluster membership** from ``clusters.json``:
word/name-labeled or ``curated`` clusters with ≥2 members that resolve to
real tags/annotations.

For each retained member span:
  * **mel** — production ``cluster_sounds.log_mel_embed`` (required)
  * **mel_norm** — analysis RMS + no-duration variant (optional, default on)
  * **HuBERT** — mean-pooled ``facebook/hubert-base-ls960`` if deps available
    (``--skip-hubert`` to disable)

Reports within-cluster vs between-cluster pairwise cosine distances, gap
(= mean_between − mean_within), and per-cluster tightness / duration /
speaker purity. SHORT fragments and non-verbal categories are excluded by
default (matching Clustering product practice).

Outputs under ``tools/analysis/out/curated_clusters/``:
  summary.json, report.md, optional hist/UMAP PNGs.

Usage
-----
  tools/.venv/bin/python tools/analysis/curated_cluster_learn.py
  tools/.venv/bin/python tools/analysis/curated_cluster_learn.py --skip-hubert
  tools/.venv/bin/python tools/analysis/curated_cluster_learn.py \\
      --library ~/Documents/BabyTalk/Library --out tools/analysis/out/curated_clusters
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from babytalk_paths import LIBRARY_DIR, list_local_kits  # noqa: E402
from cluster_sounds import (  # noqa: E402
    collect_spans,
    fragment_cutoff_ms,
    is_likely_fragment,
    log_mel_embed,
    resolve_audio,
    speaker_bucket,
)

# Reuse analysis embed helpers / plots from sibling module.
from embed_compare_mel_hubert import (  # noqa: E402
    HIST_BINS,
    HIST_XLIM,
    HUBERT_MODEL_ID,
    HUBERT_SR,
    MAX_DIFF_PAIRS_HIST,
    NONVERBAL_CATEGORY,
    hubert_embed_batch,
    log_mel_embed_analysis,
    maybe_subsample,
    pairwise_same_diff,
    plot_histograms,
    plot_umap,
    resample_to,
    rms_normalize,
    slice_samples,
)

OUT_DIR = Path(__file__).resolve().parent / "out" / "curated_clusters"

# Prior word-string study (embed_compare_norm, mel_norm_on excl nonverbal)
PRIOR_WORD_STRING_GAP_MEL_NORM = 0.1543
PRIOR_WORD_STRING_GAP_HUBERT = 0.0949


def _require_core_deps() -> None:
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
    if missing:
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + f"\n  tools/.venv/bin/pip install {' '.join(missing)}\n"
        )


def cluster_label(cl: dict) -> str | None:
    """Human-facing label for a curated cluster (word → phonetic → note)."""
    for field in ("word", "phonetic", "note"):
        raw = cl.get(field)
        if raw is None:
            continue
        val = str(raw).strip()
        if val and val.lower() != "untitled":
            return val
    return None


def cluster_label_key(cl: dict) -> str | None:
    lab = cluster_label(cl)
    return lab.lower() if lab else None


def is_curated_or_labeled(cl: dict) -> bool:
    if cl.get("curated"):
        return True
    for key in ("word", "phonetic", "language", "category", "note"):
        v = cl.get(key)
        if isinstance(v, str) and v.strip():
            return True
        if v and not isinstance(v, str):
            return True
    return False


def cluster_category(cl: dict) -> str:
    return str(cl.get("category") or "").strip().lower()


def is_nonverbal_cluster(cl: dict) -> bool:
    return cluster_category(cl) == NONVERBAL_CATEGORY


def load_tags_by_uuid(kit: Path) -> dict[str, dict]:
    """uuid → tag/annotation record (for category/speaker fallback)."""
    out: dict[str, dict] = {}
    if (kit / "tags.json").exists():
        data = json.loads((kit / "tags.json").read_text(encoding="utf-8"))
        items = data.get("tags", data if isinstance(data, list) else [])
        for t in items:
            uid = (t.get("uuid") or "").strip()
            if uid:
                out[uid] = t
    if (kit / "annotations.json").exists():
        data = json.loads((kit / "annotations.json").read_text(encoding="utf-8"))
        items = data.get("annotations", data if isinstance(data, list) else [])
        for a in items:
            if a.get("status") == "dismissed":
                continue
            uid = (a.get("uuid") or "").strip()
            if uid and uid not in out:
                out[uid] = a
    return out


def discover_curated_clusters(
    library: Path,
    *,
    min_members: int = 2,
    exclude_nonverbal: bool = True,
) -> list[dict]:
    """Scan kits for curated/labeled multi-member clusters."""
    found: list[dict] = []
    for kit in list_local_kits(library):
        path = kit / "clusters.json"
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] skip {kit.name}: bad clusters.json ({e})", flush=True)
            continue
        clusters = doc.get("clusters") or []
        if not isinstance(clusters, list):
            continue
        for cl in clusters:
            if not is_curated_or_labeled(cl):
                continue
            members = [m for m in (cl.get("members") or []) if m.get("memberId")]
            if len(members) < min_members:
                continue
            lab = cluster_label(cl)
            # Prefer word-like: require a word/phonetic/note label when possible.
            # Still keep curated=True multi-member with only category if labeled.
            if not lab and not cl.get("curated"):
                continue
            if exclude_nonverbal and is_nonverbal_cluster(cl):
                continue
            found.append(
                {
                    "kit": kit.name,
                    "kit_path": kit,
                    "cluster_id": cl.get("id") or "",
                    "word": cl.get("word"),
                    "phonetic": cl.get("phonetic"),
                    "category": cl.get("category"),
                    "note": cl.get("note"),
                    "curated": bool(cl.get("curated")),
                    "label": lab or "(unnamed curated)",
                    "label_key": cluster_label_key(cl) or f"id:{cl.get('id')}",
                    "members_raw": members,
                    "n_raw": len(members),
                }
            )
    return found


def stats_1d(xs: np.ndarray) -> dict:
    if xs is None or len(xs) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(len(xs)),
        "mean": float(np.mean(xs)),
        "median": float(np.median(xs)),
        "std": float(np.std(xs)),
        "min": float(np.min(xs)),
        "max": float(np.max(xs)),
    }


def speaker_purity(speakers: list[str]) -> dict:
    if not speakers:
        return {"purity": None, "majority": None, "n_speakers": 0, "counts": {}}
    counts = Counter(speakers)
    majority, maj_n = counts.most_common(1)[0]
    return {
        "purity": float(maj_n / len(speakers)),
        "majority": majority,
        "n_speakers": len(counts),
        "counts": dict(counts),
    }


def within_cluster_mean(D: np.ndarray, idxs: list[int]) -> float | None:
    if len(idxs) < 2:
        return None
    vals = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            vals.append(float(D[idxs[a], idxs[b]]))
    return float(np.mean(vals)) if vals else None


def build_member_rows(
    curated: list[dict],
    *,
    exclude_short: bool,
    exclude_nonverbal_members: bool,
) -> tuple[list[dict], dict]:
    """Resolve members to audio spans; apply SHORT / nonverbal filters."""
    rows: list[dict] = []
    filter_stats = {
        "clusters_in": len(curated),
        "members_raw": 0,
        "dropped_short": 0,
        "dropped_missing_span": 0,
        "dropped_nonverbal_tag": 0,
        "dropped_no_audio": 0,
        "kept": 0,
        "clusters_kept": 0,
        "fragment_cutoffs_ms": {},
    }

    # Group by kit for span index + cutoff.
    by_kit: dict[str, list[dict]] = {}
    for c in curated:
        by_kit.setdefault(c["kit"], []).append(c)

    for kit_name, kit_clusters in by_kit.items():
        kit: Path = kit_clusters[0]["kit_path"]
        spans = collect_spans(kit)
        span_by_id = {s["memberId"]: s for s in spans}
        tags_by_uuid = load_tags_by_uuid(kit)
        cutoff = fragment_cutoff_ms(spans) if exclude_short else None
        filter_stats["fragment_cutoffs_ms"][kit_name] = cutoff

        audio_path = resolve_audio(kit)
        if not audio_path.exists():
            print(f"[warn] no audio for {kit_name}", flush=True)
            for c in kit_clusters:
                filter_stats["members_raw"] += c["n_raw"]
                filter_stats["dropped_no_audio"] += c["n_raw"]
            continue

        import soundfile as sf

        audio, sr = sf.read(str(audio_path), always_2d=False)
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        for c in kit_clusters:
            kept_local: list[dict] = []
            for m in c["members_raw"]:
                filter_stats["members_raw"] += 1
                mid = m["memberId"]
                sp = span_by_id.get(mid)
                if sp is None:
                    # Fall back to member timing if present on the cluster record.
                    if m.get("startMs") is None or m.get("endMs") is None:
                        filter_stats["dropped_missing_span"] += 1
                        continue
                    sp = {
                        "memberId": mid,
                        "startMs": int(m["startMs"]),
                        "endMs": int(m["endMs"]),
                        "speaker": m.get("speaker"),
                        "speakerCluster": m.get("speakerCluster"),
                        "refType": m.get("refType") or "tag",
                        "uuid": m.get("uuid") or mid,
                    }
                if exclude_short and cutoff is not None and is_likely_fragment(sp, cutoff):
                    filter_stats["dropped_short"] += 1
                    continue
                tag = tags_by_uuid.get(mid) or tags_by_uuid.get(sp.get("uuid") or "")
                cat = ""
                if tag:
                    cat = str(tag.get("category") or "").strip().lower()
                if not cat:
                    cat = cluster_category(
                        {"category": c.get("category")}
                    )
                if exclude_nonverbal_members and cat == NONVERBAL_CATEGORY:
                    filter_stats["dropped_nonverbal_tag"] += 1
                    continue
                start_ms = float(sp["startMs"])
                end_ms = float(sp["endMs"])
                clip = slice_samples(audio, sr, start_ms, end_ms)
                if clip is None:
                    filter_stats["dropped_missing_span"] += 1
                    continue
                dur_ms = max(0.0, end_ms - start_ms)
                row = {
                    "kit": kit_name,
                    "cluster_id": c["cluster_id"],
                    "cluster_key": f"{kit_name}::{c['cluster_id']}",
                    "label": c["label"],
                    "label_key": c["label_key"],
                    "category": c.get("category"),
                    "member_id": mid,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "dur_ms": dur_ms,
                    "speaker": speaker_bucket(sp),
                    "clip": clip,
                    "sr": int(sr),
                }
                kept_local.append(row)
            if len(kept_local) >= 2:
                filter_stats["clusters_kept"] += 1
                filter_stats["kept"] += len(kept_local)
                rows.extend(kept_local)
            else:
                # Cluster fell below min members after filters.
                filter_stats["kept"] += 0

    return rows, filter_stats


def embed_rows(
    rows: list[dict],
    *,
    do_mel_norm: bool,
    do_hubert: bool,
) -> dict[str, np.ndarray]:
    """Compute embedding matrices aligned to ``rows`` order."""
    mel = []
    mel_norm = []
    for i, r in enumerate(rows):
        mel.append(log_mel_embed(r["clip"], r["sr"]))
        if do_mel_norm:
            mel_norm.append(
                log_mel_embed_analysis(r["clip"], r["sr"], normalize=True)
            )
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[mel] {i + 1}/{len(rows)}", flush=True)
    out: dict[str, np.ndarray] = {"mel": np.vstack(mel)}
    if do_mel_norm:
        out["mel_norm"] = np.vstack(mel_norm)

    if do_hubert:
        clips_16k = []
        for r in rows:
            x = r["clip"]
            # Light RMS normalize helps HuBERT a bit; analysis-only.
            x = rms_normalize(x)
            clips_16k.append(resample_to(x, r["sr"], HUBERT_SR))
        print(f"[hubert] embedding {len(clips_16k)} clips…", flush=True)
        try:
            out["hubert"] = hubert_embed_batch(clips_16k, batch_size=8, device="cpu")
        except Exception as e:
            print(f"[hubert] failed ({e}); continuing without HuBERT", flush=True)

    return out


def gap_table_row(name: str, same: np.ndarray, diff: np.ndarray) -> dict:
    mean_same = float(np.mean(same)) if len(same) else None
    mean_diff = float(np.mean(diff)) if len(diff) else None
    gap = (
        float(mean_diff - mean_same)
        if mean_same is not None and mean_diff is not None
        else None
    )
    return {
        "space": name,
        "mean_within": mean_same,
        "mean_between": mean_diff,
        "gap": gap,
        "n_within": int(len(same)),
        "n_between": int(len(diff)),
        "within": stats_1d(same),
        "between": stats_1d(diff),
    }


def per_cluster_stats(
    rows: list[dict],
    embeddings: np.ndarray,
) -> list[dict]:
    from sklearn.metrics.pairwise import cosine_distances

    D = cosine_distances(embeddings)
    # index groups
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["cluster_key"], []).append(i)

    out = []
    for key, idxs in groups.items():
        r0 = rows[idxs[0]]
        durs = np.asarray([rows[i]["dur_ms"] for i in idxs], dtype=np.float64)
        speakers = [rows[i]["speaker"] for i in idxs]
        within = within_cluster_mean(D, idxs)
        out.append(
            {
                "cluster_key": key,
                "kit": r0["kit"],
                "cluster_id": r0["cluster_id"],
                "label": r0["label"],
                "label_key": r0["label_key"],
                "category": r0.get("category"),
                "n_members": len(idxs),
                "mean_within_mel": within,
                "duration_ms": {
                    "median": float(np.median(durs)),
                    "min": float(np.min(durs)),
                    "max": float(np.max(durs)),
                    "mean": float(np.mean(durs)),
                },
                "speaker": speaker_purity(speakers),
            }
        )
    out.sort(
        key=lambda x: (
            x["mean_within_mel"] is None,
            x["mean_within_mel"] if x["mean_within_mel"] is not None else 9e9,
        )
    )
    return out


def try_plot_umap(embeddings: np.ndarray, labels: list[str], out_path: Path, title: str) -> bool:
    try:
        import umap  # noqa: F401
    except ImportError:
        print("[umap] umap-learn not installed — skipping UMAP", flush=True)
        return False
    try:
        plot_umap(embeddings, labels, out_path, title=title)
        return True
    except Exception as e:
        print(f"[umap] failed: {e}", flush=True)
        return False


def write_report(
    path: Path,
    *,
    summary: dict,
) -> None:
    gap_rows = summary.get("gap_table") or []
    clusters = summary.get("clusters_ranked") or []
    filt = summary.get("filters") or {}
    disc = summary.get("discovery") or {}

    lines: list[str] = []
    lines.append("# Curated cluster reverse-engineering")
    lines.append("")
    lines.append(
        "Acoustic distances on **manually curated / labeled** clusters "
        "(grouped annotated words), not raw same-`word` string pairs and not "
        "auto-only unlabeled junk clusters."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"- Kits with curated multi-member clusters kept: "
        f"**{disc.get('n_kits_kept', '?')}** / {disc.get('n_kits_scanned', '?')} scanned "
        f"({', '.join(disc.get('kits_kept') or []) or 'none'})"
    )
    lines.append(
        f"- Clusters (raw curated ≥2): **{disc.get('n_clusters_raw', '?')}**; "
        f"after filters (≥2 members): **{disc.get('n_clusters_kept', '?')}**"
    )
    lines.append(
        f"- Members: raw **{filt.get('members_raw', '?')}** → kept **{filt.get('kept', '?')}** "
        f"(dropped short={filt.get('dropped_short', 0)}, "
        f"nonverbal={filt.get('dropped_nonverbal_tag', 0)}, "
        f"missing={filt.get('dropped_missing_span', 0)})"
    )
    lines.append("")
    lines.append("## Filters (documented)")
    lines.append("")
    lines.append(
        "1. **Curated/labeled only** — `curated=true` or non-empty "
        "`word`/`phonetic`/`language`/`category`/`note`; require ≥2 members."
    )
    lines.append(
        "2. **Non-verbal clusters excluded by default** — cluster "
        f"`category == '{NONVERBAL_CATEGORY}'` dropped (prefer word-like groups)."
    )
    lines.append(
        "3. **SHORT fragments excluded by default** — kit-adaptive cutoff "
        "(median tag duration × 0.55, clamped 400–500 ms), same as Clustering "
        "tab SHORT badge / `cluster_sounds.fragment_cutoff_ms`."
    )
    lines.append(
        "4. Membership must resolve to a loadable audio span "
        "(`tags.json` / `annotations.json` via `collect_spans`)."
    )
    cut = filt.get("fragment_cutoffs_ms") or {}
    if cut:
        lines.append("")
        lines.append("Per-kit SHORT cutoffs (ms):")
        for k, v in cut.items():
            lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Gap table (pairwise cosine distance)")
    lines.append("")
    lines.append(
        "Within = pairs sharing the same curated cluster id; "
        "between = pairs from different curated clusters. "
        "Gap = mean_between − mean_within (larger is better separation)."
    )
    lines.append("")
    lines.append("| Space | mean within | mean between | gap | n_within | n_between |")
    lines.append("|-------|------------:|-------------:|----:|---------:|----------:|")
    for r in gap_rows:
        mw = r.get("mean_within")
        mb = r.get("mean_between")
        g = r.get("gap")
        mw_s = f"{mw:.4f}" if mw is not None else "—"
        mb_s = f"{mb:.4f}" if mb is not None else "—"
        g_s = f"**{g:.4f}**" if g is not None else "—"
        lines.append(
            f"| `{r['space']}` | {mw_s} | {mb_s} | {g_s} | "
            f"{r.get('n_within', 0)} | {r.get('n_between', 0)} |"
        )
    lines.append("")
    prior = summary.get("prior_comparison") or {}
    if prior:
        lines.append("### vs earlier word-string study")
        lines.append("")
        lines.append(
            "Earlier `embed_compare` used same-`word` string pairs on hand tags "
            f"(mel_norm excl nonverbal gap ≈ **{PRIOR_WORD_STRING_GAP_MEL_NORM:.4f}**; "
            f"HuBERT ≈ **{PRIOR_WORD_STRING_GAP_HUBERT:.4f}**)."
        )
        if prior.get("mel_norm_gap") is not None:
            delta = prior["mel_norm_gap"] - PRIOR_WORD_STRING_GAP_MEL_NORM
            lines.append(
                f"- This curated-id mel_norm gap: **{prior['mel_norm_gap']:.4f}** "
                f"(Δ vs word-string = {delta:+.4f})."
            )
        if prior.get("hubert_gap") is not None:
            delta_h = prior["hubert_gap"] - PRIOR_WORD_STRING_GAP_HUBERT
            lines.append(
                f"- This curated-id HuBERT gap: **{prior['hubert_gap']:.4f}** "
                f"(Δ vs word-string = {delta_h:+.4f})."
            )
        lines.append("")

    tight = [c for c in clusters if c.get("mean_within_mel") is not None][:8]
    loose = [c for c in reversed(clusters) if c.get("mean_within_mel") is not None][:8]

    lines.append("## Tightest curated clusters (mel within)")
    lines.append("")
    lines.append("| Rank | Label | Kit | n | mean within | med dur ms | speaker purity |")
    lines.append("|-----:|-------|-----|--:|------------:|-----------:|---------------:|")
    for i, c in enumerate(tight, 1):
        sp = (c.get("speaker") or {}).get("purity")
        sp_s = f"{sp:.2f}" if sp is not None else "—"
        lines.append(
            f"| {i} | {c['label']!r} | `{c['kit']}` | {c['n_members']} | "
            f"{c['mean_within_mel']:.4f} | {c['duration_ms']['median']:.0f} | "
            f"{sp_s} |"
        )
    lines.append("")
    lines.append("## Loosest / messy curated clusters (mel within)")
    lines.append("")
    lines.append("| Rank | Label | Kit | n | mean within | med dur ms | speaker purity |")
    lines.append("|-----:|-------|-----|--:|------------:|-----------:|---------------:|")
    for i, c in enumerate(loose, 1):
        sp = (c.get("speaker") or {}).get("purity")
        sp_s = f"{sp:.2f}" if sp is not None else "—"
        lines.append(
            f"| {i} | {c['label']!r} | `{c['kit']}` | {c['n_members']} | "
            f"{c['mean_within_mel']:.4f} | {c['duration_ms']['median']:.0f} | "
            f"{sp_s} |"
        )
    lines.append("")

    messy = [
        c
        for c in clusters
        if c.get("mean_within_mel") is not None
        and (
            c["mean_within_mel"] >= 0.75
            or (c.get("speaker") or {}).get("purity", 1.0) < 0.7
            or c["duration_ms"]["max"] > 3 * max(c["duration_ms"]["median"], 1)
        )
    ]
    if messy:
        lines.append("### Call-outs (messy)")
        lines.append("")
        for c in messy[:12]:
            reasons = []
            if c["mean_within_mel"] >= 0.75:
                reasons.append(f"loose within={c['mean_within_mel']:.3f}")
            pur = (c.get("speaker") or {}).get("purity")
            if pur is not None and pur < 0.7:
                reasons.append(f"mixed speakers purity={pur:.2f}")
            if c["duration_ms"]["max"] > 3 * max(c["duration_ms"]["median"], 1):
                reasons.append(
                    f"dur spread {c['duration_ms']['min']:.0f}–{c['duration_ms']['max']:.0f} ms"
                )
            lines.append(
                f"- **{c['label']}** (`{c['kit']}`, n={c['n_members']}): "
                + "; ".join(reasons)
            )
        lines.append("")

    plots = summary.get("plots") or {}
    if plots:
        lines.append("## Plots")
        lines.append("")
        for name, p in plots.items():
            lines.append(f"- `{name}`: `{p}`")
        lines.append("")

    lines.append("## Conclusions for Clustering product")
    lines.append("")
    for bullet in summary.get("takeaways") or []:
        lines.append(f"- {bullet}")
    lines.append("")
    lines.append("---")
    lines.append("")
    emb = summary.get("embed") or {}
    lines.append(
        f"Generated by `tools/analysis/curated_cluster_learn.py`. "
        f"Embed: {emb.get('mel')}; distance={emb.get('distance')}; "
        f"same-key={emb.get('same_key')}."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_takeaways(summary: dict) -> list[str]:
    gaps = {r["space"]: r for r in (summary.get("gap_table") or [])}
    clusters = summary.get("clusters_ranked") or []
    filt = summary.get("filters") or {}
    mel = gaps.get("mel") or gaps.get("mel_prod")
    mel_n = gaps.get("mel_norm")
    hub = gaps.get("hubert")

    out: list[str] = []
    g_mel = (mel_n or mel or {}).get("gap")
    if g_mel is not None:
        vs = g_mel - PRIOR_WORD_STRING_GAP_MEL_NORM
        out.append(
            f"Curated-id mel gap is {g_mel:.3f} "
            f"({'wider' if vs > 0.02 else 'similar to' if abs(vs) <= 0.02 else 'narrower than'} "
            f"word-string study {PRIOR_WORD_STRING_GAP_MEL_NORM:.3f}, Δ={vs:+.3f}) — "
            "human cluster membership agrees with same-word tags on mel separation; "
            "use curated IDs as the cleaner training target when both exist."
        )
    if hub and hub.get("gap") is not None and g_mel is not None:
        if hub["gap"] < g_mel:
            out.append(
                f"HuBERT gap ({hub['gap']:.3f}) is smaller than mel "
                f"({g_mel:.3f}); keep production mel fingerprints as the "
                "default clustering embed — HuBERT is not worth the cost here."
            )
        else:
            out.append(
                f"HuBERT gap ({hub['gap']:.3f}) beats mel ({g_mel:.3f}); "
                "consider optional HuBERT for ranking/merge suggestions, "
                "not as a hard default until latency is acceptable."
            )

    tight = [c for c in clusters if c.get("mean_within_mel") is not None][:5]
    if tight:
        names = ", ".join(f"{c['label']!r} ({c['mean_within_mel']:.2f})" for c in tight[:4])
        out.append(f"Tightest curated words (low within): {names}.")

    loose = [c for c in clusters if c.get("mean_within_mel") is not None]
    loose = sorted(loose, key=lambda c: -c["mean_within_mel"])[:4]
    if loose:
        names = ", ".join(f"{c['label']!r} ({c['mean_within_mel']:.2f})" for c in loose)
        out.append(
            f"Messiest curated groups (high within / review candidates): {names}."
        )

    dropped_short = filt.get("dropped_short") or 0
    kept = filt.get("kept") or 0
    if dropped_short:
        out.append(
            f"SHORT filter removed {dropped_short} member(s) "
            f"({100 * dropped_short / max(dropped_short + kept, 1):.0f}% of "
            "raw curated members before keep); keep SHORT excluded when "
            "seeding auto-clusters — curated truth still leans word-length."
        )
    else:
        out.append(
            "Few/no SHORT members appeared inside curated multi-member "
            "clusters after filters — reviewers already grouped word-like spans; "
            "continue excluding shorts from auto clustering."
        )

    mixed = [
        c
        for c in clusters
        if (c.get("speaker") or {}).get("purity") is not None
        and (c.get("speaker") or {}).get("purity", 1) < 0.85
    ]
    if mixed:
        out.append(
            f"{len(mixed)} curated cluster(s) have speaker purity < 0.85 — "
            "product should keep auto-clusters speaker-homogeneous, but allow "
            "explicit curated overrides when humans intentionally mix."
        )
    else:
        out.append(
            "Curated clusters are largely speaker-pure — reinforces keeping "
            "auto-clustering speaker-bucketed while locking curated groups intact."
        )
    return out[:5]


def main() -> int:
    _require_core_deps()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--library",
        type=Path,
        default=LIBRARY_DIR,
        help=f"BabyTalk Library (default: {LIBRARY_DIR})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="Output directory",
    )
    ap.add_argument("--min-members", type=int, default=2)
    ap.add_argument(
        "--include-nonverbal",
        action="store_true",
        help="Keep non-verbal vocalization curated clusters",
    )
    ap.add_argument(
        "--include-short",
        action="store_true",
        help="Keep SHORT fragment members (default: exclude)",
    )
    ap.add_argument(
        "--skip-mel-norm",
        action="store_true",
        help="Skip analysis mel_norm embeddings",
    )
    ap.add_argument(
        "--skip-hubert",
        action="store_true",
        help="Skip HuBERT (mel only)",
    )
    ap.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip histogram / UMAP PNGs",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    library = args.library.expanduser()
    out_dir = args.out
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_nv = not args.include_nonverbal
    exclude_short = not args.include_short

    print(f"[scan] library={library}", flush=True)
    curated = discover_curated_clusters(
        library,
        min_members=args.min_members,
        exclude_nonverbal=exclude_nv,
    )
    kits_scanned = sum(
        1 for k in list_local_kits(library) if (k / "clusters.json").exists()
    )
    print(
        f"[scan] kits_with_clusters={kits_scanned} "
        f"curated_multi={len(curated)}",
        flush=True,
    )
    for c in curated:
        print(
            f"  · {c['kit']}: {c['label']!r} n={c['n_raw']} "
            f"curated={c['curated']} cat={c.get('category')!r}",
            flush=True,
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
        f"{filter_stats['clusters_kept']} clusters "
        f"(raw members={filter_stats['members_raw']})",
        flush=True,
    )
    if filter_stats["clusters_kept"] < 2 or filter_stats["kept"] < 4:
        raise SystemExit(
            "Too few members/clusters after filters to score between-cluster gap."
        )

    do_mel_norm = not args.skip_mel_norm
    do_hubert = not args.skip_hubert
    if do_hubert:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            print("[hubert] deps missing — skipping", flush=True)
            do_hubert = False

    embeds = embed_rows(rows, do_mel_norm=do_mel_norm, do_hubert=do_hubert)
    cluster_keys = [r["cluster_key"] for r in rows]

    rng = np.random.default_rng(args.seed)
    gap_table = []
    hist_payload = {}

    for name, mat in embeds.items():
        same, diff = pairwise_same_diff(mat, cluster_keys)
        gap_table.append(gap_table_row(name, same, diff))
        hist_payload[name] = (same, maybe_subsample(diff, MAX_DIFF_PAIRS_HIST, rng))
        print(
            f"[gap] {name}: within={np.mean(same):.4f} between={np.mean(diff):.4f} "
            f"gap={np.mean(diff) - np.mean(same):.4f} "
            f"(n_w={len(same)} n_b={len(diff)})",
            flush=True,
        )

    clusters_ranked = per_cluster_stats(rows, embeds["mel"])
    # Attach mel_norm within if present
    if "mel_norm" in embeds:
        from sklearn.metrics.pairwise import cosine_distances

        D_n = cosine_distances(embeds["mel_norm"])
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(r["cluster_key"], []).append(i)
        by_key = {c["cluster_key"]: c for c in clusters_ranked}
        for key, idxs in groups.items():
            by_key[key]["mean_within_mel_norm"] = within_cluster_mean(D_n, idxs)

    kits_kept = sorted({r["kit"] for r in rows})
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

    plots: dict[str, str] = {}
    if not args.skip_plots:
        import matplotlib

        matplotlib.use("Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
        (out_dir / ".mplconfig").mkdir(exist_ok=True)

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(Path.cwd()))
            except ValueError:
                return str(p)

        # Histogram: mel vs hubert or mel vs mel_norm
        mel_same, mel_diff = hist_payload.get("mel", (np.array([]), np.array([])))
        if "hubert" in hist_payload:
            h_same, h_diff = hist_payload["hubert"]
            hist_path = out_dir / "hist_mel_vs_hubert.png"
            plot_histograms(
                mel_same,
                mel_diff,
                h_same,
                h_diff,
                hist_path,
                title="Curated clusters: within vs between",
                mel_title="Mel (production)",
                hub_title="HuBERT",
            )
            plots["hist_mel_vs_hubert"] = _rel(hist_path)
        elif "mel_norm" in hist_payload:
            n_same, n_diff = hist_payload["mel_norm"]
            hist_path = out_dir / "hist_mel_vs_mel_norm.png"
            plot_histograms(
                mel_same,
                mel_diff,
                n_same,
                n_diff,
                hist_path,
                title="Curated clusters: within vs between",
                mel_title="Mel (production)",
                hub_title="Mel norm (analysis)",
            )
            plots["hist_mel_vs_mel_norm"] = _rel(hist_path)
        else:
            # Single-panel hist
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(
                mel_same,
                bins=HIST_BINS,
                density=True,
                alpha=0.65,
                label="within",
                color="#2a6f97",
            )
            ax.hist(
                mel_diff,
                bins=HIST_BINS,
                density=True,
                alpha=0.55,
                label="between",
                color="#bc4749",
            )
            ax.set_xlim(*HIST_XLIM)
            ax.set_xlabel("cosine distance")
            ax.set_ylabel("density")
            ax.set_title("Mel (production) curated within vs between")
            ax.legend()
            fig.tight_layout()
            hist_path = out_dir / "hist_mel.png"
            fig.savefig(hist_path, dpi=140)
            plt.close(fig)
            plots["hist_mel"] = _rel(hist_path)

        umap_path = out_dir / "umap_mel.png"
        if try_plot_umap(
            embeds["mel"],
            [r["label"] for r in rows],
            umap_path,
            title="Curated clusters — mel UMAP",
        ):
            plots["umap_mel"] = _rel(umap_path)

        if "hubert" in embeds:
            umap_h = out_dir / "umap_hubert.png"
            if try_plot_umap(
                embeds["hubert"],
                [r["label"] for r in rows],
                umap_h,
                title="Curated clusters — HuBERT UMAP",
            ):
                plots["umap_hubert"] = _rel(umap_h)

    prior_comparison = {
        "word_string_mel_norm_gap": PRIOR_WORD_STRING_GAP_MEL_NORM,
        "word_string_hubert_gap": PRIOR_WORD_STRING_GAP_HUBERT,
        "mel_norm_gap": next(
            (r["gap"] for r in gap_table if r["space"] == "mel_norm"), None
        ),
        "mel_gap": next((r["gap"] for r in gap_table if r["space"] == "mel"), None),
        "hubert_gap": next(
            (r["gap"] for r in gap_table if r["space"] == "hubert"), None
        ),
    }

    summary = {
        "mode": "curated_cluster_membership",
        "library": str(library),
        "discovery": discovery,
        "filters": {
            **filter_stats,
            "exclude_short": exclude_short,
            "exclude_nonverbal": exclude_nv,
            "short_rule": (
                "kit-adaptive median_tag_dur×0.55 clamped [400,500] ms "
                "(cluster_sounds.fragment_cutoff_ms); matches Clustering SHORT badge"
            ),
            "nonverbal_rule": f"category == '{NONVERBAL_CATEGORY}'",
            "curated_rule": (
                "curated=true OR non-empty word/phonetic/language/category/note; "
                f"≥{args.min_members} members; prefer human word/phonetic/note label"
            ),
        },
        "embed": {
            "mel": "cluster_sounds.log_mel_embed (production)",
            "mel_norm": (
                "analysis RMS→0.1 + fixed grid + per-bin z-score, no duration"
                if do_mel_norm
                else None
            ),
            "hubert": HUBERT_MODEL_ID if "hubert" in embeds else None,
            "distance": "sklearn cosine_distances (= 1 − cosine similarity)",
            "same_key": "curated cluster id (kit::cluster_id), not raw word string",
        },
        "gap_table": gap_table,
        "prior_comparison": prior_comparison,
        "clusters_ranked": clusters_ranked,
        "plots": plots,
        "takeaways": [],
    }
    summary["takeaways"] = build_takeaways(summary)

    # JSON-safe: drop non-serializable if any
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path = out_dir / "report.md"
    write_report(report_path, summary=summary)

    print(f"\n[done] {summary_path}", flush=True)
    print(f"[done] {report_path}", flush=True)
    for t in summary["takeaways"]:
        print(f"  • {t}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
