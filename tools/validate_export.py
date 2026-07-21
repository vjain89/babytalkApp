"""Validate BabyTalk session kits (schema + basic coverage).

Usage:
  python3 tools/validate_export.py /path/to/Backups/2026-07-20_1200
  python3 tools/validate_export.py /path/to/single_kit_folder
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_MANIFEST = {
    "schemaVersion",
    "recordingUuid",
    "audioFile",
    "durationMs",
    "audioContentHash",
    "codec",
    "sampleRate",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_kit(kit: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = kit / "manifest.json"
    if not manifest_path.exists():
        return [f"{kit.name}: missing manifest.json"]

    try:
        manifest = load_json(manifest_path)
    except Exception as e:
        return [f"{kit.name}: bad manifest.json ({e})"]

    missing = REQUIRED_MANIFEST - set(manifest)
    if missing:
        errors.append(f"{kit.name}: manifest missing {sorted(missing)}")

    audio_name = manifest.get("audioFile", "audio.wav")
    audio_path = kit / audio_name
    if not audio_path.exists():
        errors.append(f"{kit.name}: missing audio file {audio_name}")
    elif audio_path.stat().st_size < 100:
        errors.append(f"{kit.name}: audio file suspiciously small")

    duration = manifest.get("durationMs") or 0
    if duration <= 0:
        errors.append(f"{kit.name}: durationMs should be > 0")

    tags_path = kit / "tags.json"
    tag_count = 0
    if tags_path.exists():
        tags_payload = load_json(tags_path)
        tags = tags_payload.get("tags", tags_payload if isinstance(tags_payload, list) else [])
        tag_count = len(tags)
        for t in tags:
            if "uuid" not in t or "label" not in t:
                errors.append(f"{kit.name}: tag missing uuid/label")
                break

    ann_path = kit / "annotations.json"
    ann_count = 0
    if ann_path.exists():
        ann_payload = load_json(ann_path)
        anns = ann_payload.get("annotations", ann_payload if isinstance(ann_payload, list) else [])
        ann_count = len(anns)

    density = tag_count / (duration / 60000.0) if duration > 0 else 0
    print(
        f"OK  {kit.name}: duration={duration/1000:.1f}s tags={tag_count} "
        f"annotations={ann_count} tags_per_min={density:.2f}"
    )
    return errors


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    root = Path(argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"Path not found: {root}")
        return 1

    kits: list[Path] = []
    if (root / "manifest.json").exists():
        kits = [root]
    else:
        kits = sorted([p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()])
        if not kits:
            print(f"No session kits found under {root}")
            return 1

    all_errors: list[str] = []
    for kit in kits:
        all_errors.extend(validate_kit(kit))

    if all_errors:
        print("\nErrors:")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print(f"\nValidated {len(kits)} kit(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
