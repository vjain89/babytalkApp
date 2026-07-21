"""Fixed Mac paths for BabyTalk review + USB sync."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HOME = Path.home()
BABYTALK_ROOT = HOME / "Documents" / "BabyTalk"
LIBRARY_DIR = BABYTALK_ROOT / "Library"
SYNC_STATE_PATH = BABYTALK_ROOT / ".sync-state.json"
MOUNT_HINTS_DIR = BABYTALK_ROOT / ".cache"

# Dev / File Sharing bundle id (PRODUCT_NAME = babytalkApp).
DEFAULT_BUNDLE_ID = "org.reactjs.native.example.babytalkApp"


def ensure_library() -> Path:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    BABYTALK_ROOT.mkdir(parents=True, exist_ok=True)
    return LIBRARY_DIR


def load_sync_state() -> dict:
    if not SYNC_STATE_PATH.exists():
        return {"kits": {}, "lastSyncAt": None}
    try:
        with SYNC_STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"kits": {}, "lastSyncAt": None}
        data.setdefault("kits", {})
        return data
    except Exception:
        return {"kits": {}, "lastSyncAt": None}


def save_sync_state(state: dict) -> None:
    ensure_library()
    with SYNC_STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def list_local_kits(library: Path | None = None) -> list[Path]:
    root = library or ensure_library()
    if (root / "manifest.json").exists():
        return [root]
    return sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()
    )


def seed_library_from_backups(extra_roots: list[Path] | None = None) -> int:
    """
    If Library is empty, copy kits from known Mac backup trees into Library/
    (flat: one folder per kit). Returns number of kits copied.
    """
    ensure_library()
    if list_local_kits():
        return 0

    candidates: list[Path] = []
    backups = BABYTALK_ROOT / "Backups"
    if backups.exists():
        candidates.append(backups)
    for root in extra_roots or []:
        if root.exists():
            candidates.append(root)

    copied = 0
    for batch_root in candidates:
        for path in batch_root.rglob("manifest.json"):
            kit = path.parent
            # Skip Library itself if nested somehow
            try:
                kit.relative_to(LIBRARY_DIR)
                continue
            except ValueError:
                pass
            dest = LIBRARY_DIR / kit.name
            if dest.exists():
                continue
            shutil.copytree(kit, dest)
            copied += 1
    return copied
