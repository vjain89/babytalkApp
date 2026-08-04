"""Library-wide curated vocabulary tightness for the Review Browser.

Ranks human-labeled / curated word clusters by mel within-cluster mean cosine
distance (same definition as ``analysis/curated_cluster_learn.py``), merging
the same normalized word label across kits into one row.

Embedding cache + daily rank history live under ``tools/analysis/out/``.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = TOOLS_DIR / "analysis"
OUT_DIR = ANALYSIS_DIR / "out" / "vocab_tightness"
HISTORY_PATH = ANALYSIS_DIR / "out" / "cluster_tightness_history.json"
EMBED_CACHE_PATH = OUT_DIR / "embeddings_cache.npz"

# Keep recent day-rank columns manageable in the table.
MAX_HISTORY_DAY_COLUMNS = 14

_lock = threading.Lock()
_status: dict = {
    "busy": False,
    "phase": "idle",
    "done": 0,
    "total": 0,
    "error": None,
    "message": "",
    "updated_at": None,
}


def get_status() -> dict:
    with _lock:
        return dict(_status)


def _set_status(**kwargs) -> None:
    with _lock:
        _status.update(kwargs)
        _status["updated_at"] = datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return date.today().isoformat()


def load_history(path: Path | None = None) -> dict:
    p = path or HISTORY_PATH
    if not p.exists():
        return {"version": 1, "days": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "days": []}
    if not isinstance(data, dict):
        return {"version": 1, "days": []}
    data.setdefault("version", 1)
    days = data.get("days")
    if not isinstance(days, list):
        data["days"] = []
    return data


def save_history(hist: dict, path: Path | None = None) -> None:
    p = path or HISTORY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hist, indent=2) + "\n", encoding="utf-8")


def _load_embed_cache() -> dict[str, np.ndarray]:
    if not EMBED_CACHE_PATH.exists():
        return {}
    try:
        raw = np.load(str(EMBED_CACHE_PATH), allow_pickle=False)
        keys_raw = raw["keys"]
        # keys stored as fixed-width unicode / bytes
        if keys_raw.dtype.kind in ("U", "S"):
            keys = [str(k) for k in keys_raw.tolist()]
        else:
            keys = [str(k) for k in keys_raw.tolist()]
        vecs = raw["embeddings"]
        return {k: vecs[i] for i, k in enumerate(keys)}
    except Exception:
        return {}


def _save_embed_cache(cache: dict[str, np.ndarray]) -> None:
    if not cache:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache.keys())
    embeddings = np.vstack([cache[k] for k in keys])
    # Unicode array avoids pickle on load.
    key_arr = np.asarray(keys, dtype=f"U{max(8, max(len(k) for k in keys))}")
    np.savez_compressed(
        str(EMBED_CACHE_PATH),
        keys=key_arr,
        embeddings=embeddings,
    )


def _member_cache_key(row: dict) -> str:
    return (
        f"{row['kit']}|{row['member_id']}|"
        f"{int(round(row['start_ms']))}|{int(round(row['end_ms']))}"
    )


def library_clusters_fingerprint(library: Path) -> str:
    """Cheap fingerprint of all kits' clusters.json (mtime + size).

    Used to invalidate the cached ranking when the user curates new clusters
    without forcing a full Refresh.
    """
    import hashlib

    from babytalk_paths import list_local_kits

    h = hashlib.sha1()
    for kit in list_local_kits(library):
        path = kit / "clusters.json"
        if not path.exists():
            continue
        try:
            st = path.stat()
            h.update(f"{kit.name}:{st.st_mtime_ns}:{st.st_size}\n".encode())
        except OSError:
            continue
    return h.hexdigest()[:16]


def _session_name(kit_path: Path) -> str | None:
    man = kit_path / "manifest.json"
    if not man.exists():
        return None
    try:
        data = json.loads(man.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("sessionName", "originalSessionName", "name"):
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _kit_display(kit_name: str, library: Path | None = None) -> str:
    """Prefer human session name (e.g. 26_07_27__19:53:00) over folder suffix."""
    if library is not None:
        sess = _session_name(library / kit_name)
        if sess:
            return sess[:56]
    parts = kit_name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])[:48]
    return kit_name[:48]


def compute_vocab_ranking(
    library: Path,
    *,
    exclude_short: bool = True,
    exclude_nonverbal: bool = True,
    progress: bool = True,
) -> dict:
    """Discover curated clusters, embed (cached), merge by label, rank tight→loose."""
    # Import lazily so review_server can start without analysis deps loaded.
    import sys

    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    from curated_cluster_learn import (  # type: ignore
        build_member_rows,
        discover_curated_clusters,
        within_cluster_mean,
    )
    from cluster_sounds import log_mel_embed
    from sklearn.metrics.pairwise import cosine_distances

    if progress:
        _set_status(busy=True, phase="discover", done=0, total=0, error=None, message="Scanning kits…")

    curated = discover_curated_clusters(
        library,
        min_members=2,
        exclude_nonverbal=exclude_nonverbal,
    )
    rows, filt = build_member_rows(
        curated,
        exclude_short=exclude_short,
        exclude_nonverbal_members=exclude_nonverbal,
    )

    if progress:
        _set_status(
            phase="embed",
            done=0,
            total=len(rows),
            message=f"Embedding {len(rows)} segments…",
        )

    cache = _load_embed_cache()
    embeddings: list[np.ndarray] = []
    dirty = False
    for i, row in enumerate(rows):
        key = _member_cache_key(row)
        vec = cache.get(key)
        if vec is None:
            vec = log_mel_embed(row["clip"], row["sr"])
            cache[key] = vec
            dirty = True
        embeddings.append(np.asarray(vec, dtype=np.float64))
        if progress and ((i + 1) % 10 == 0 or i + 1 == len(rows)):
            _set_status(
                phase="embed",
                done=i + 1,
                total=len(rows),
                message=f"Embedded {i + 1}/{len(rows)}",
            )

    if dirty:
        if progress:
            _set_status(phase="cache", message="Saving embedding cache…")
        try:
            _save_embed_cache(cache)
        except Exception as e:
            # Cache is best-effort; ranking still proceeds.
            if progress:
                _set_status(message=f"Cache save failed ({e}); continuing")

    if progress:
        _set_status(phase="rank", message="Ranking merged clusters…")

    # Drop heavy clip arrays before grouping.
    light_rows = []
    for r, emb in zip(rows, embeddings):
        light_rows.append(
            {
                "kit": r["kit"],
                "cluster_id": r["cluster_id"],
                "label": r["label"],
                "label_key": r["label_key"],
                "category": r.get("category"),
                "member_id": r["member_id"],
                "start_ms": float(r["start_ms"]),
                "end_ms": float(r["end_ms"]),
                "dur_ms": float(r["dur_ms"]),
                "speaker": r.get("speaker"),
                "embedding": emb,
            }
        )

    groups: dict[str, list[dict]] = {}
    for r in light_rows:
        groups.setdefault(r["label_key"], []).append(r)

    clusters: list[dict] = []
    for label_key, members in groups.items():
        # Drop exact duplicate time spans (same kit + window) — often a
        # double-add in clusters.json that collapses tightness to 0.
        deduped: list[dict] = []
        seen_span: set[tuple] = set()
        for m in members:
            span_key = (
                m["kit"],
                int(round(m["start_ms"])),
                int(round(m["end_ms"])),
            )
            if span_key in seen_span:
                continue
            seen_span.add(span_key)
            deduped.append(m)
        members = deduped
        if len(members) < 2:
            continue
        mat = np.vstack([m["embedding"] for m in members])
        D = cosine_distances(mat)
        idxs = list(range(len(members)))
        tightness = within_cluster_mean(D, idxs)
        kits = sorted({m["kit"] for m in members})
        kit_clusters: dict[str, dict] = {}
        for m in members:
            kc = kit_clusters.setdefault(
                m["kit"],
                {
                    "kit": m["kit"],
                    "kit_display": _kit_display(m["kit"], library),
                    "cluster_id": m["cluster_id"],
                    "label": m["label"],
                    "members": [],
                },
            )
            # Prefer first non-empty cluster_id / label if mixed.
            if not kc.get("cluster_id") and m["cluster_id"]:
                kc["cluster_id"] = m["cluster_id"]
            kc["members"].append(
                {
                    "memberId": m["member_id"],
                    "startMs": int(round(m["start_ms"])),
                    "endMs": int(round(m["end_ms"])),
                    "durMs": int(round(m["dur_ms"])),
                    "speaker": m.get("speaker") or "?",
                    "audioUrl": f"/audio?kit={m['kit']}",
                }
            )
        label = members[0]["label"]
        # Prefer a real word-like label over id: fallbacks.
        for m in members:
            if m["label"] and not str(m["label_key"]).startswith("id:"):
                label = m["label"]
                break
        clusters.append(
            {
                "label": label,
                "label_key": label_key,
                "tightness": tightness,
                "n_members": len(members),
                "n_kits": len(kits),
                "kits": kits,
                "kit_clusters": list(kit_clusters.values()),
                "category": members[0].get("category"),
            }
        )

    clusters.sort(
        key=lambda c: (
            c["tightness"] is None,
            c["tightness"] if c["tightness"] is not None else 9e9,
            c["label_key"],
        )
    )
    for i, c in enumerate(clusters, start=1):
        c["rank"] = i

    result = {
        "ok": True,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "date": _today(),
        "library": str(library),
        "source_fingerprint": library_clusters_fingerprint(library),
        "n_clusters": len(clusters),
        "clusters": clusters,
        "filters": filt,
        "embed": {
            "space": "cluster_sounds.log_mel_embed",
            "distance": "sklearn cosine_distances (mean pairwise within cluster)",
            "cache": str(EMBED_CACHE_PATH),
        },
        "history_path": str(HISTORY_PATH),
    }
    if progress:
        _set_status(
            busy=False,
            phase="done",
            done=len(rows),
            total=len(rows),
            message=f"{len(clusters)} clusters ranked",
            error=None,
        )
    return result


def snapshot_from_result(result: dict) -> dict:
    ranks = {}
    tightness = {}
    for c in result.get("clusters") or []:
        key = c["label_key"]
        ranks[key] = c.get("rank")
        tightness[key] = c.get("tightness")
    return {
        "date": result.get("date") or _today(),
        "recorded_at": result.get("computed_at") or datetime.now(timezone.utc).isoformat(),
        "n_clusters": result.get("n_clusters") or len(result.get("clusters") or []),
        "ranks": ranks,
        "tightness": tightness,
    }


def upsert_today_snapshot(hist: dict, result: dict, *, replace: bool = False) -> dict:
    """Append or replace today's day entry. Returns updated history."""
    snap = snapshot_from_result(result)
    days = hist.setdefault("days", [])
    today = snap["date"]
    idx = next((i for i, d in enumerate(days) if d.get("date") == today), None)
    if idx is None:
        days.append(snap)
    elif replace:
        days[idx] = snap
    # else keep existing today entry (idempotent load)
    days.sort(key=lambda d: d.get("date") or "")
    hist["days"] = days
    return hist


def history_has_today(hist: dict | None = None) -> bool:
    h = hist if hist is not None else load_history()
    today = _today()
    return any(d.get("date") == today for d in h.get("days") or [])


def attach_history_columns(
    result: dict,
    hist: dict,
    *,
    max_days: int = MAX_HISTORY_DAY_COLUMNS,
    prior_ranks: dict | None = None,
) -> dict:
    """Add ``history_days`` + per-cluster ``rank_by_day`` / ``rank_delta`` for the UI.

    ``prior_ranks`` is optional (label_key → rank) from the snapshot we are
    replacing — used when only one history day exists so mid-day refreshes
    can still show ↑/↓ / new.
    """
    days = list(hist.get("days") or [])
    # Most recent N days, chronological left→right.
    recent = days[-max_days:] if len(days) > max_days else days
    day_dates = [d.get("date") for d in recent if d.get("date")]
    prev_ranks: dict = {}
    if prior_ranks is not None:
        prev_ranks = prior_ranks
    elif len(days) >= 2:
        prev_ranks = days[-2].get("ranks") or {}

    for c in result.get("clusters") or []:
        key = c["label_key"]
        by_day = {}
        for d in recent:
            dt = d.get("date")
            if not dt:
                continue
            ranks = d.get("ranks") or {}
            by_day[dt] = ranks.get(key)  # None → missing that day
        c["rank_by_day"] = by_day
        # When history has only today, keep deltas already baked into a cached
        # ranking file (mid-day refresh vs the snapshot we replaced).
        if prior_ranks is None and len(days) < 2 and ("rank_delta" in c or "is_new" in c):
            continue
        cur = c.get("rank")
        old = prev_ranks.get(key)
        if not prev_ranks:
            c["rank_delta"] = None
            c["is_new"] = False
        elif cur is None or old is None:
            c["rank_delta"] = None
            c["is_new"] = old is None
        else:
            # Positive delta = improved (rank number decreased).
            c["rank_delta"] = int(old) - int(cur)
            c["is_new"] = False
    series = [
        {"date": d.get("date"), "n_clusters": d.get("n_clusters") or 0}
        for d in days
        if d.get("date")
    ]
    result["history_days"] = day_dates
    result["history_series"] = series
    result["history_path"] = str(HISTORY_PATH)
    result["has_today"] = history_has_today(hist)
    return result


def ensure_ranking(
    library: Path,
    *,
    force: bool = False,
    exclude_short: bool = True,
    exclude_nonverbal: bool = True,
) -> dict:
    """
    Return ranked clusters + history.

    - If today is already snapshotted and not ``force``, recompute is skipped only
      when the cached last-result matches today's date *and* the clusters.json
      fingerprint is unchanged; otherwise we recompute (embeddings stay cached).
    - Always persists today's snapshot when missing; ``force`` or a fingerprint
      change replaces today's entry so the chart / ranks stay current.
    """
    hist = load_history()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    last_path = OUT_DIR / "last_ranking.json"
    fp_now = library_clusters_fingerprint(library)

    # Capture prior ranks before we overwrite today's snapshot (for Δ column).
    prior_ranks: dict | None = None
    today_entry = next(
        (d for d in (hist.get("days") or []) if d.get("date") == _today()),
        None,
    )
    if today_entry and isinstance(today_entry.get("ranks"), dict):
        prior_ranks = dict(today_entry["ranks"])
    elif hist.get("days"):
        last_day = hist["days"][-1]
        if isinstance(last_day.get("ranks"), dict):
            prior_ranks = dict(last_day["ranks"])

    need_compute = force or not history_has_today(hist) or not last_path.exists()
    if not need_compute and last_path.exists():
        try:
            cached = json.loads(last_path.read_text(encoding="utf-8"))
            same_day = cached.get("date") == _today()
            same_fp = cached.get("source_fingerprint") == fp_now
            if same_day and same_fp and cached.get("clusters") is not None:
                attach_history_columns(cached, hist, prior_ranks=None)
                cached["from_cache"] = True
                cached["snapshot_written"] = False
                _set_status(
                    busy=False,
                    phase="done",
                    message=f"{cached.get('n_clusters', len(cached.get('clusters') or []))} clusters (cached)",
                    error=None,
                )
                return cached
            # Curated clusters changed since last compute → refresh.
            need_compute = True
        except Exception:
            need_compute = True

    try:
        result = compute_vocab_ranking(
            library,
            exclude_short=exclude_short,
            exclude_nonverbal=exclude_nonverbal,
            progress=True,
        )
    except Exception as e:
        _set_status(
            busy=False,
            phase="error",
            error=str(e),
            message=str(e),
        )
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()[-2000:],
            "history_path": str(HISTORY_PATH),
            "status": get_status(),
        }

    # Fresh compute always upserts today's snapshot (append or replace) so the
    # chart and daily rank columns stay aligned with current curated clusters.
    hist = upsert_today_snapshot(hist, result, replace=True)
    save_history(hist)
    wrote = True

    # Always refresh last_ranking for detail panel / fast reload.
    try:
        last_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    hist = load_history()
    # Prefer yesterday for Δ when we have ≥2 days; else use pre-replace today.
    if len(hist.get("days") or []) >= 2:
        attach_history_columns(result, hist, prior_ranks=None)
    else:
        attach_history_columns(result, hist, prior_ranks=prior_ranks)
    result["from_cache"] = False
    result["snapshot_written"] = wrote
    return result


def start_ensure_async(library: Path, *, force: bool = False) -> dict:
    """Kick off ensure_ranking in a daemon thread if not already busy."""
    with _lock:
        if _status.get("busy"):
            return {"ok": True, "started": False, "status": dict(_status)}
        _status["busy"] = True
        _status["phase"] = "starting"
        _status["error"] = None
        _status["message"] = "Starting…"

    def _run() -> None:
        try:
            ensure_ranking(library, force=force)
        except Exception as e:
            _set_status(busy=False, phase="error", error=str(e), message=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "started": True, "status": get_status()}
