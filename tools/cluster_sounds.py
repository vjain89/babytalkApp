"""Acoustic clustering of BabyTalk spans within a session kit (cluster_v0).

Collects confirmed tags + non-dismissed annotations, fingerprints each clip
(log-mel summary), groups similar sounds, and writes ``clusters.json``.

Cluster labels (word / phonetic / language / category) live on the cluster
only — members stay linked by id for training and future concepts (M:N).

Requires: numpy, soundfile, scikit-learn
  tools/.venv/bin/pip install numpy soundfile scikit-learn

Usage:
  python3 tools/cluster_sounds.py ~/Documents/BabyTalk/Library/<kit>
  python3 tools/cluster_sounds.py <kit> --distance 0.55
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install deps: pip install numpy soundfile\n" + str(e)) from e

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install scikit-learn: tools/.venv/bin/pip install scikit-learn\n" + str(e)
    ) from e

SCHEMA_VERSION = 1
SOURCE = "cluster_v0"
# Cosine distance threshold for agglomerative clustering (lower = tighter groups).
# Log-mel temporal fingerprints need a fairly low threshold to split word-like units.
DEFAULT_DISTANCE = 0.45
MIN_DUR_MS = 120
PAD_MS = 40
N_MELS = 40
N_FFT = 512
HOP = 160
EMBED_CACHE = "cluster_embeddings.npz"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def resolve_audio(kit: Path) -> Path:
    manifest = load_json(kit / "manifest.json")
    path = kit / manifest.get("audioFile", "audio.wav")
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    return path


def collect_spans(kit: Path) -> list[dict]:
    """Tags + non-dismissed annotations as clusterable spans."""
    spans: list[dict] = []
    seen: set[str] = set()

    if (kit / "tags.json").exists():
        tags = load_json(kit / "tags.json")
        items = tags.get("tags", tags if isinstance(tags, list) else [])
        for t in items:
            uid = (t.get("uuid") or "").strip()
            if not uid or uid in seen:
                continue
            start = int(t.get("startMs") if t.get("startMs") is not None else 0)
            end = t.get("endMs")
            if end is None:
                end = start + 300
            end = int(end)
            if end < start:
                start, end = end, start
            if end - start < MIN_DUR_MS:
                continue
            seen.add(uid)
            spans.append(
                {
                    "memberId": uid,
                    "refType": "tag",
                    "uuid": uid,
                    "startMs": start,
                    "endMs": end,
                    "speaker": (t.get("speaker") or "").strip() or None,
                    "source": t.get("source") or "user",
                }
            )

    if (kit / "annotations.json").exists():
        anns = load_json(kit / "annotations.json")
        items = anns.get("annotations", anns if isinstance(anns, list) else [])
        for a in items:
            if a.get("status") == "dismissed":
                continue
            uid = (a.get("uuid") or "").strip()
            if not uid or uid in seen:
                continue
            start = int(
                a.get("startMs")
                if a.get("startMs") is not None
                else a.get("tMs") or 0
            )
            end = a.get("endMs")
            if end is None:
                end = start + 300
            end = int(end)
            if end < start:
                start, end = end, start
            if end - start < MIN_DUR_MS:
                continue
            seen.add(uid)
            spans.append(
                {
                    "memberId": uid,
                    "refType": "annotation",
                    "uuid": uid,
                    "startMs": start,
                    "endMs": end,
                    "speaker": (a.get("speaker") or "").strip() or None,
                    "source": a.get("source") or "vad_v0",
                    "status": a.get("status") or "provisional",
                }
            )
    return spans


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_max = sr / 2.0
    m_min, m_max = _hz_to_mel(0.0), _hz_to_mel(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    hz_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (j - left) / max(1, center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (right - j) / max(1, right - center)
    return fb


def log_mel_embed(
    samples: np.ndarray,
    sr: int,
    *,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop: int = HOP,
    n_frames_out: int = 32,
) -> np.ndarray:
    """L2-normalized fingerprint: fixed-grid log-mel spectrogram (time×mel).

    Resamples variable-length clips onto a fixed frame grid so shape differences
    (not just average color) drive clustering.
    """
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float64)
    # Light peak normalize so loudness doesn't dominate.
    peak = float(np.max(np.abs(samples))) + 1e-12
    samples = samples / peak
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
    # Interpolate to fixed time grid.
    t_old = np.linspace(0.0, 1.0, num=len(mat))
    t_new = np.linspace(0.0, 1.0, num=n_frames_out)
    fixed = np.zeros((n_frames_out, n_mels), dtype=np.float64)
    for j in range(n_mels):
        fixed[:, j] = np.interp(t_new, t_old, mat[:, j])
    # Per-bin z-score across the clip reduces channel bias.
    fixed = fixed - fixed.mean(axis=0, keepdims=True)
    std = fixed.std(axis=0, keepdims=True) + 1e-8
    fixed = fixed / std
    feat = fixed.reshape(-1)
    dur = math.log10(max(len(samples), 1) / sr * 1000.0)
    feat = np.concatenate([feat, np.array([dur], dtype=np.float64)])
    norm = np.linalg.norm(feat) + 1e-12
    return (feat / norm).astype(np.float64)


def slice_embed(
    audio: np.ndarray,
    sr: int,
    start_ms: int,
    end_ms: int,
    *,
    pad_ms: int = PAD_MS,
) -> np.ndarray:
    start = max(0, int((start_ms - pad_ms) * sr / 1000))
    end = min(len(audio), int((end_ms + pad_ms) * sr / 1000))
    if end <= start:
        end = min(len(audio), start + int(0.2 * sr))
    return log_mel_embed(audio[start:end], sr)


def mel_matrix_for_slice(
    audio: np.ndarray,
    sr: int,
    start_ms: int,
    end_ms: int,
    *,
    pad_ms: int = PAD_MS,
    max_frames: int = 80,
) -> list[list[float]]:
    """Compact log-mel for UI spectrogram comparator (list of frames)."""
    start = max(0, int((start_ms - pad_ms) * sr / 1000))
    end = min(len(audio), int((end_ms + pad_ms) * sr / 1000))
    samples = np.asarray(audio[start:end], dtype=np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if len(samples) < N_FFT:
        samples = np.pad(samples, (0, N_FFT - len(samples)))
    window = np.hanning(N_FFT)
    fb = _mel_filterbank(sr, N_FFT, N_MELS)
    frames = []
    for s in range(0, max(1, len(samples) - N_FFT + 1), HOP):
        frame = samples[s : s + N_FFT] * window
        spec = np.abs(np.fft.rfft(frame)) ** 2
        mel = np.log(fb @ spec + 1e-10)
        frames.append(mel.tolist())
        if len(frames) >= max_frames:
            break
    return frames


def cluster_labels(embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(embeddings)


def compute_confidence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
) -> dict:
    """Four review signals: tightness, size, separation, outliers."""
    idx = np.where(labels == cluster_id)[0]
    size = int(len(idx))
    members = embeddings[idx]
    centroid = members.mean(axis=0)
    cnorm = np.linalg.norm(centroid) + 1e-12
    centroid = centroid / cnorm

    # Cosine distance to centroid for each member.
    sims = members @ centroid
    dists = 1.0 - sims
    mean_d = float(dists.mean()) if size else 1.0
    # Tightness: 1 = identical, 0 = orthogonal/far.
    tightness = float(max(0.0, min(1.0, 1.0 - mean_d)))

    # Separation: distance from this centroid to nearest other centroid.
    other_ids = sorted(set(labels.tolist()) - {cluster_id})
    if not other_ids:
        separation = 1.0
    else:
        other_cents = []
        for oid in other_ids:
            o = embeddings[labels == oid].mean(axis=0)
            o = o / (np.linalg.norm(o) + 1e-12)
            other_cents.append(o)
        other_cents = np.stack(other_cents, axis=0)
        sep_d = 1.0 - (other_cents @ centroid)
        separation = float(np.min(sep_d))
        separation = max(0.0, min(1.0, separation))

    # Outliers: farther than mean + 1.5 std within cluster (need ≥3 members).
    outlier_local: list[int] = []
    if size >= 3:
        thr = mean_d + 1.5 * float(dists.std())
        for local_i, d in enumerate(dists):
            if float(d) > thr:
                outlier_local.append(int(local_i))
    elif size == 2 and mean_d > 0.45:
        # Flag the farther of a loose pair.
        outlier_local = [int(np.argmax(dists))]

    return {
        "tightness": round(tightness, 4),
        "size": size,
        "separation": round(separation, 4),
        "outlierMemberIndexes": outlier_local,
        "meanDistanceToCentroid": round(mean_d, 4),
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def preserve_cluster_identity(
    new_clusters: list[dict],
    old_doc: dict | None,
    *,
    min_jaccard: float = 0.4,
) -> list[dict]:
    """Reuse old cluster ids + labels when member sets still overlap."""
    if not old_doc:
        return new_clusters
    old_list = old_doc.get("clusters") or []
    used_old: set[str] = set()
    for cl in new_clusters:
        new_ids = {m["memberId"] for m in cl.get("members") or []}
        best, best_j = None, 0.0
        for old in old_list:
            oid = old.get("id")
            if not oid or oid in used_old:
                continue
            old_ids = {m["memberId"] for m in old.get("members") or []}
            j = _jaccard(new_ids, old_ids)
            if j > best_j:
                best_j, best = j, old
        if best and best_j >= min_jaccard:
            used_old.add(best["id"])
            cl["id"] = best["id"]
            for key in ("word", "phonetic", "language", "category", "note"):
                if best.get(key) and not cl.get(key):
                    cl[key] = best[key]
            cl["preservedFrom"] = {"jaccard": round(best_j, 3)}
    return new_clusters


def _cluster_is_locked(cl: dict) -> bool:
    """Labeled or hand-curated clusters must survive re-runs intact."""
    if cl.get("curated"):
        return True
    for key in ("word", "phonetic", "language", "category", "note"):
        if (cl.get(key) or "").strip() if isinstance(cl.get(key), str) else cl.get(key):
            return True
    return False


def _backup_clusters_file(kit: Path) -> Path | None:
    path = kit / "clusters.json"
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = kit / f"clusters.json.bak-{stamp}"
    bak.write_bytes(path.read_bytes())
    # Keep a stable pointer to the latest backup.
    latest = kit / "clusters.json.bak"
    latest.write_bytes(path.read_bytes())
    # Prune old stamped backups (keep 8).
    stamped = sorted(kit.glob("clusters.json.bak-*"), reverse=True)
    for old in stamped[8:]:
        try:
            old.unlink()
        except OSError:
            pass
    return bak


def process_kit(
    kit: Path,
    *,
    distance_threshold: float = DEFAULT_DISTANCE,
    write: bool = True,
    include_singletons: bool = True,
) -> dict:
    kit = kit.resolve()
    if not (kit / "manifest.json").exists():
        return {"ok": False, "error": "no manifest", "kit": kit.name}

    spans = collect_spans(kit)
    if not spans:
        doc = {
            "schemaVersion": SCHEMA_VERSION,
            "source": SOURCE,
            "kit": kit.name,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "params": {"distanceThreshold": distance_threshold},
            "clusters": [],
            "unclustered": [],
        }
        if write:
            write_json(kit / "clusters.json", doc)
        return {"ok": True, "kit": kit.name, "clusters": 0, "spans": 0, "doc": doc}

    old_doc = None
    if (kit / "clusters.json").exists():
        try:
            old_doc = load_json(kit / "clusters.json")
        except (json.JSONDecodeError, OSError):
            old_doc = None

    # Lock hand-labeled / curated clusters completely; only regroup free spans.
    locked_clusters: list[dict] = []
    locked_ids: set[str] = set()
    if old_doc:
        for cl in old_doc.get("clusters") or []:
            if not _cluster_is_locked(cl):
                continue
            members = [dict(m) for m in (cl.get("members") or []) if m.get("memberId")]
            if not members:
                continue
            kept = dict(cl)
            kept["members"] = members
            kept["locked"] = True
            locked_clusters.append(kept)
            for m in members:
                locked_ids.add(m["memberId"])

    free_spans = [sp for sp in spans if sp["memberId"] not in locked_ids]
    audio_path = resolve_audio(kit)
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float64)
    sr = int(sr)

    # Embeddings for all spans (cache); clustering only on free spans.
    all_embeds = []
    for sp in spans:
        all_embeds.append(slice_embed(audio, sr, sp["startMs"], sp["endMs"]))
    X_all = np.stack(all_embeds, axis=0) if spans else np.zeros((0, 1))
    span_index = {sp["memberId"]: i for i, sp in enumerate(spans)}

    new_clusters: list[dict] = []
    if free_spans:
        free_idx = [span_index[sp["memberId"]] for sp in free_spans]
        X = X_all[np.array(free_idx)]
        labels = cluster_labels(X, distance_threshold)
        for cid in sorted(set(labels.tolist())):
            idx = np.where(labels == cid)[0]
            conf = compute_confidence(X, labels, int(cid))
            members = []
            outlier_idxs = set(conf.pop("outlierMemberIndexes"))
            for local_i, gi in enumerate(idx):
                m = dict(free_spans[int(gi)])
                m["outlier"] = local_i in outlier_idxs
                members.append(m)
            members.sort(key=lambda m: m["startMs"])
            if not include_singletons and len(members) < 2:
                continue
            new_clusters.append(
                {
                    "id": str(uuid.uuid4()),
                    "members": members,
                    "confidence": conf,
                    "word": None,
                    "phonetic": None,
                    "language": None,
                    "category": None,
                    "note": None,
                    "conceptIds": [],
                    "curated": False,
                }
            )

    # Refresh confidence on locked clusters from cache vectors when possible.
    for cl in locked_clusters:
        conf = cl.setdefault("confidence", {})
        conf["size"] = len(cl.get("members") or [])
        idxs = [
            span_index[m["memberId"]]
            for m in (cl.get("members") or [])
            if m.get("memberId") in span_index
        ]
        if len(idxs) >= 1 and len(X_all):
            # Separation vs other locked+new is approximate; tightness from members.
            sub = X_all[np.array(idxs)]
            centroid = sub.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            dists = 1.0 - (sub @ centroid)
            mean_d = float(dists.mean())
            conf["tightness"] = round(float(max(0.0, min(1.0, 1.0 - mean_d))), 4)
            conf["meanDistanceToCentroid"] = round(mean_d, 4)
            conf.setdefault("separation", 0.0)

    clusters = locked_clusters + new_clusters
    clusters.sort(
        key=lambda c: (
            -(c.get("confidence") or {}).get("size", 0),
            -(c.get("confidence") or {}).get("tightness", 0),
            -(c.get("confidence") or {}).get("separation", 0),
        )
    )

    if write:
        _backup_clusters_file(kit)
        np.savez_compressed(
            kit / EMBED_CACHE,
            member_ids=np.array([s["memberId"] for s in spans]),
            embeddings=X_all,
        )

    doc = {
        "schemaVersion": SCHEMA_VERSION,
        "source": SOURCE,
        "kit": kit.name,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "params": {
            "distanceThreshold": distance_threshold,
            "minDurMs": MIN_DUR_MS,
            "embed": "logmel_fixedgrid_v1",
            "preserve": "locked_labeled_or_curated",
        },
        "clusters": clusters,
        "spanCount": len(spans),
        "lockedClusterCount": len(locked_clusters),
        "conceptsNote": "Concepts (developmental threads) are not stored here yet; cluster.conceptIds reserved.",
    }
    if write:
        write_json(kit / "clusters.json", doc)

    return {
        "ok": True,
        "kit": kit.name,
        "spans": len(spans),
        "clusters": len(clusters),
        "multiMember": sum(1 for c in clusters if (c.get("confidence") or {}).get("size", 0) >= 2),
        "lockedClusters": len(locked_clusters),
        "freeSpans": len(free_spans),
        "backup": True,
        "doc": doc,
    }


def update_cluster_labels(kit: Path, cluster_id: str, fields: dict) -> dict:
    path = kit / "clusters.json"
    if not path.exists():
        return {"ok": False, "error": "clusters.json missing — run clustering first"}
    doc = load_json(path)
    found = None
    for cl in doc.get("clusters") or []:
        if cl.get("id") == cluster_id:
            found = cl
            break
    if not found:
        return {"ok": False, "error": f"cluster not found: {cluster_id}"}
    for key in ("word", "phonetic", "language", "category", "note"):
        if key in fields:
            val = fields.get(key)
            if val is None or str(val).strip() == "":
                found[key] = None
            else:
                found[key] = str(val).strip()
    doc["updatedAt"] = datetime.now(timezone.utc).isoformat()
    found["curated"] = True
    write_json(path, doc)
    return {"ok": True, "cluster": found}


def exclude_member(kit: Path, cluster_id: str, member_id: str) -> dict:
    path = kit / "clusters.json"
    if not path.exists():
        return {"ok": False, "error": "clusters.json missing"}
    doc = load_json(path)
    clusters = doc.get("clusters") or []
    target = None
    for cl in clusters:
        if cl.get("id") == cluster_id:
            target = cl
            break
    if not target:
        return {"ok": False, "error": "cluster not found"}
    before = len(target.get("members") or [])
    target["members"] = [
        m for m in (target.get("members") or []) if m.get("memberId") != member_id
    ]
    if len(target["members"]) == before:
        return {"ok": False, "error": "member not in cluster"}
    target["confidence"]["size"] = len(target["members"])
    target["curated"] = True
    # Drop empty clusters.
    doc["clusters"] = [c for c in clusters if (c.get("members") or [])]
    doc["updatedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(path, doc)
    return {"ok": True, "clusterId": cluster_id, "memberId": member_id}


def _refresh_confidence_from_cache(kit: Path, cluster: dict) -> None:
    """Best-effort confidence update using cluster_embeddings.npz if present."""
    cache_path = kit / EMBED_CACHE
    conf = cluster.setdefault("confidence", {})
    members = cluster.get("members") or []
    conf["size"] = len(members)
    if not cache_path.exists() or len(members) == 0:
        conf.setdefault("tightness", 0.0)
        conf.setdefault("separation", 0.0)
        return
    try:
        cache = np.load(cache_path, allow_pickle=True)
        ids = [str(x) for x in cache["member_ids"].tolist()]
        X = cache["embeddings"]
        id_to_i = {mid: i for i, mid in enumerate(ids)}
        idxs = [id_to_i[m["memberId"]] for m in members if m.get("memberId") in id_to_i]
        if len(idxs) < 1:
            return
        # Fake labels array for compute_confidence: all same id 0 among subset.
        # Separation vs other clusters needs full doc — approximate with tightness only.
        sub = X[np.array(idxs)]
        centroid = sub.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = sub @ centroid
        dists = 1.0 - sims
        mean_d = float(dists.mean())
        conf["tightness"] = round(float(max(0.0, min(1.0, 1.0 - mean_d))), 4)
        conf["meanDistanceToCentroid"] = round(mean_d, 4)
        outlier_local: list[int] = []
        if len(idxs) >= 3:
            thr = mean_d + 1.5 * float(dists.std())
            outlier_local = [i for i, d in enumerate(dists) if float(d) > thr]
        for i, m in enumerate(members):
            m["outlier"] = i in outlier_local
        # Keep previous separation if present; mark stale lightly.
        conf.setdefault("separation", 0.0)
    except Exception:  # noqa: BLE001
        return


def merge_clusters(kit: Path, keep_id: str, merge_id: str) -> dict:
    """Absorb ``merge_id`` into ``keep_id`` (members + labels fill-empty)."""
    keep_id = (keep_id or "").strip()
    merge_id = (merge_id or "").strip()
    if not keep_id or not merge_id:
        return {"ok": False, "error": "keepId and mergeId required"}
    if keep_id == merge_id:
        return {"ok": False, "error": "cannot merge a cluster into itself"}
    path = kit / "clusters.json"
    if not path.exists():
        return {"ok": False, "error": "clusters.json missing"}
    doc = load_json(path)
    clusters = doc.get("clusters") or []
    keep = next((c for c in clusters if c.get("id") == keep_id), None)
    other = next((c for c in clusters if c.get("id") == merge_id), None)
    if not keep or not other:
        return {"ok": False, "error": "cluster not found"}

    by_id = {m["memberId"]: m for m in (keep.get("members") or []) if m.get("memberId")}
    for m in other.get("members") or []:
        mid = m.get("memberId")
        if mid and mid not in by_id:
            by_id[mid] = m
    keep["members"] = sorted(by_id.values(), key=lambda m: m.get("startMs") or 0)

    for key in ("word", "phonetic", "language", "category", "note"):
        if not keep.get(key) and other.get(key):
            keep[key] = other[key]
    concepts = list(keep.get("conceptIds") or [])
    for cid in other.get("conceptIds") or []:
        if cid not in concepts:
            concepts.append(cid)
    keep["conceptIds"] = concepts
    keep["curated"] = True

    doc["clusters"] = [c for c in clusters if c.get("id") != merge_id]
    _refresh_confidence_from_cache(kit, keep)
    doc["updatedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(path, doc)
    return {"ok": True, "cluster": keep, "mergedAway": merge_id}


def _span_record_for_uuid(kit: Path, member_id: str) -> dict | None:
    """Build a cluster member record from tags.json or annotations.json."""
    member_id = (member_id or "").strip()
    if not member_id:
        return None
    for sp in collect_spans(kit):
        if sp.get("memberId") == member_id:
            return dict(sp)
    return None


def add_member_to_cluster(kit: Path, cluster_id: str, member_id: str) -> dict:
    """Manually pull a tag/annotation span into a cluster (and detach elsewhere)."""
    cluster_id = (cluster_id or "").strip()
    member_id = (member_id or "").strip()
    if not cluster_id or not member_id:
        return {"ok": False, "error": "clusterId and memberId required"}
    path = kit / "clusters.json"
    if not path.exists():
        return {"ok": False, "error": "clusters.json missing — run clustering first"}
    doc = load_json(path)
    clusters = doc.get("clusters") or []
    target = next((c for c in clusters if c.get("id") == cluster_id), None)
    if not target:
        return {"ok": False, "error": "cluster not found"}

    record = _span_record_for_uuid(kit, member_id)
    if not record:
        return {
            "ok": False,
            "error": "span not found in tags/annotations (dismissed VAD excluded)",
        }

    # Remove from any other cluster so membership is unique.
    for cl in clusters:
        if cl.get("id") == cluster_id:
            continue
        before = len(cl.get("members") or [])
        cl["members"] = [
            m for m in (cl.get("members") or []) if m.get("memberId") != member_id
        ]
        if len(cl.get("members") or []) != before:
            conf = cl.setdefault("confidence", {})
            conf["size"] = len(cl["members"])

    members = target.setdefault("members", [])
    if any(m.get("memberId") == member_id for m in members):
        return {"ok": True, "alreadyMember": True, "cluster": target}

    members.append(record)
    members.sort(key=lambda m: m.get("startMs") or 0)
    target["curated"] = True
    doc["clusters"] = [c for c in clusters if (c.get("members") or [])]
    _refresh_confidence_from_cache(kit, target)
    doc["updatedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(path, doc)
    return {"ok": True, "cluster": target, "added": member_id}


def create_cluster_from_members(kit: Path, member_ids: list, fields: dict | None = None) -> dict:
    """Create a curated cluster from unassigned (or move-from-elsewhere) member ids."""
    fields = fields or {}
    ids = [str(x).strip() for x in (member_ids or []) if str(x).strip()]
    if not ids:
        return {"ok": False, "error": "memberIds required"}
    path = kit / "clusters.json"
    if path.exists():
        doc = load_json(path)
    else:
        doc = {
            "schemaVersion": SCHEMA_VERSION,
            "source": SOURCE,
            "kit": kit.name,
            "clusters": [],
        }

    clusters = doc.setdefault("clusters", [])
    # Detach selected ids from any existing cluster.
    id_set = set(ids)
    for cl in clusters:
        before = len(cl.get("members") or [])
        cl["members"] = [
            m for m in (cl.get("members") or []) if m.get("memberId") not in id_set
        ]
        if len(cl.get("members") or []) != before:
            cl.setdefault("confidence", {})["size"] = len(cl["members"])
            cl["curated"] = True

    members = []
    missing = []
    for mid in ids:
        rec = _span_record_for_uuid(kit, mid)
        if not rec:
            missing.append(mid)
            continue
        members.append(rec)
    if not members:
        return {"ok": False, "error": "no valid spans found", "missing": missing}
    members.sort(key=lambda m: m.get("startMs") or 0)

    cluster = {
        "id": str(uuid.uuid4()),
        "members": members,
        "confidence": {"size": len(members), "tightness": 0.0, "separation": 0.0},
        "word": None,
        "phonetic": None,
        "language": None,
        "category": None,
        "note": None,
        "conceptIds": [],
        "curated": True,
    }
    for key in ("word", "phonetic", "language", "category", "note"):
        if key in fields and fields.get(key) is not None and str(fields.get(key)).strip():
            cluster[key] = str(fields.get(key)).strip()
    _refresh_confidence_from_cache(kit, cluster)

    clusters = [c for c in clusters if (c.get("members") or [])]
    clusters.append(cluster)
    clusters.sort(
        key=lambda c: (
            -(c.get("confidence") or {}).get("size", 0),
            -(c.get("confidence") or {}).get("tightness", 0),
        )
    )
    doc["clusters"] = clusters
    doc["updatedAt"] = datetime.now(timezone.utc).isoformat()
    doc.setdefault("schemaVersion", SCHEMA_VERSION)
    doc.setdefault("source", SOURCE)
    doc["kit"] = kit.name
    write_json(path, doc)
    out = {"ok": True, "cluster": cluster}
    if missing:
        out["missing"] = missing
    return out


def promote_member_to_tag(kit: Path, cluster_id: str, member_id: str) -> dict:
    """Optional: create a tag from an annotation member; keep cluster link."""
    path = kit / "clusters.json"
    if not path.exists():
        return {"ok": False, "error": "clusters.json missing"}
    doc = load_json(path)
    cl = next((c for c in (doc.get("clusters") or []) if c.get("id") == cluster_id), None)
    if not cl:
        return {"ok": False, "error": "cluster not found"}
    member = next(
        (m for m in (cl.get("members") or []) if m.get("memberId") == member_id),
        None,
    )
    if not member:
        return {"ok": False, "error": "member not found"}
    if member.get("refType") == "tag":
        return {"ok": True, "alreadyTag": True, "uuid": member_id}

    # Build tag from annotation span; cluster labels stay on cluster only.
    tags = []
    tpath = kit / "tags.json"
    if tpath.exists():
        tp = load_json(tpath)
        tags = tp.get("tags", tp if isinstance(tp, list) else [])
    if any(t.get("uuid") == member_id for t in tags):
        member["refType"] = "tag"
        write_json(path, doc)
        return {"ok": True, "alreadyTag": True, "uuid": member_id}

    tag = {
        "uuid": member_id,
        "startMs": member["startMs"],
        "endMs": member["endMs"],
        "tMs": member["startMs"],
        "label": "untitled",
        "source": "cluster_promote",
        "status": "confirmed",
    }
    if member.get("speaker"):
        tag["speaker"] = member["speaker"]
    tags.append(tag)
    write_json(tpath, {"tags": tags})
    member["refType"] = "tag"
    member["source"] = "cluster_promote"
    doc["updatedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(path, doc)
    return {"ok": True, "promoted": True, "tag": tag}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kit", type=Path, help="Session kit folder")
    p.add_argument(
        "--distance",
        type=float,
        default=DEFAULT_DISTANCE,
        help=f"Cosine distance threshold (default {DEFAULT_DISTANCE})",
    )
    p.add_argument(
        "--no-singletons",
        action="store_true",
        help="Omit size-1 clusters from the output",
    )
    args = p.parse_args(argv)
    kit = args.kit.expanduser().resolve()
    if not kit.is_dir():
        print(f"Not a directory: {kit}")
        return 1
    result = process_kit(
        kit,
        distance_threshold=args.distance,
        write=True,
        include_singletons=not args.no_singletons,
    )
    if not result.get("ok"):
        print(json.dumps(result, indent=2))
        return 1
    slim = {k: v for k, v in result.items() if k != "doc"}
    print(json.dumps(slim, indent=2))
    print(f"Wrote {kit / 'clusters.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
