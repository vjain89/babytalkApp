#!/usr/bin/env python3
"""USB sync between the Mac BabyTalk library and the iPhone app Documents folder.

Requires:
  - iPhone plugged in, unlocked, and Trusted
  - tools/.venv with pymobiledevice3 (see tools/README.md)

Usage:
  tools/.venv/bin/python tools/iphone_sync.py status
  tools/.venv/bin/python tools/iphone_sync.py sync
  tools/.venv/bin/python tools/iphone_sync.py pull
  tools/.venv/bin/python tools/iphone_sync.py push
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from babytalk_paths import (  # noqa: E402
    DEFAULT_BUNDLE_ID,
    LIBRARY_DIR,
    ensure_library,
    list_local_kits,
    load_sync_state,
    save_sync_state,
    seed_library_from_backups,
)

try:
    from pymobiledevice3.lockdown import create_using_usbmux
    from pymobiledevice3.services.house_arrest import DOCUMENTS_ROOT, HouseArrestService
except ImportError as e:  # pragma: no cover
    create_using_usbmux = None  # type: ignore
    HouseArrestService = None  # type: ignore
    DOCUMENTS_ROOT = "/Documents"
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if limit is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit))
    return h.hexdigest()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _kit_audio_hash(kit: Path) -> str | None:
    man = kit / "manifest.json"
    if man.exists():
        try:
            m = _load_json(man)
            h = m.get("audioContentHash")
            if h:
                return str(h)
        except Exception:
            pass
    return None


def _tags_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        return None


async def _open_docs(bundle_id: str):
    if _IMPORT_ERR is not None:
        raise RuntimeError(
            "pymobiledevice3 is not installed. Run:\n"
            "  python3 -m venv tools/.venv && tools/.venv/bin/pip install -r tools/requirements.txt"
        ) from _IMPORT_ERR
    lockdown = await create_using_usbmux()
    service = await HouseArrestService.create(
        lockdown, bundle_id=bundle_id, documents_only=True
    )
    return lockdown, service


def _docs_join(*parts: str) -> str:
    """Path inside VendDocuments — rooted at /Documents."""
    cleaned = [p.strip("/") for p in parts if p and p != "/"]
    return DOCUMENTS_ROOT + ("/" + "/".join(cleaned) if cleaned else "")


async def _afc(result):
    """pymobiledevice3 AFC helpers are often async at runtime — normalize."""
    if asyncio.iscoroutine(result):
        return await result
    return result


async def _listdir(service, remote: str) -> list[str]:
    try:
        entries = await _afc(service.listdir(remote))
    except Exception:
        return []
    return [e for e in entries if e not in (".", "..")]


async def _exists(service, remote: str) -> bool:
    try:
        return bool(await _afc(service.exists(remote)))
    except Exception:
        return False


async def _isdir(service, remote: str) -> bool:
    try:
        return bool(await _afc(service.isdir(remote)))
    except Exception:
        return False


async def _makedirs(service, remote: str) -> None:
    try:
        await _afc(service.makedirs(remote))
    except Exception:
        pass


async def _get_bytes(service, remote: str) -> bytes:
    data = await _afc(service.get_file_contents(remote))
    return data if isinstance(data, (bytes, bytearray)) else bytes(data)


async def discover_device(bundle_id: str) -> dict:
    """Return connection status without transferring files."""
    out: dict = {
        "connected": False,
        "bundleId": bundle_id,
        "deviceName": None,
        "error": None,
        "hasBackups": False,
        "hasImport": False,
    }
    try:
        lockdown, service = await _open_docs(bundle_id)
    except Exception as e:
        out["error"] = str(e)
        return out
    try:
        out["connected"] = True
        try:
            out["deviceName"] = lockdown.short_info.get("DeviceName") or lockdown.display_name
        except Exception:
            out["deviceName"] = getattr(lockdown, "udid", None)
        out["hasBackups"] = await _exists(service, _docs_join("Backups"))
        out["hasImport"] = await _exists(service, _docs_join("Import"))
        return out
    finally:
        await service.close()


async def _pull_remote_file(service, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(await _get_bytes(service, remote_path))


async def _pull_remote_dir(service, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    # Prefer service.pull for trees when available
    try:
        await _afc(service.pull(remote_dir, str(local_dir), progress_bar=False))
        # pull may nest an extra folder named after the remote basename
        nested = local_dir / Path(remote_dir.rstrip("/")).name
        if nested.is_dir() and nested != local_dir:
            for child in nested.iterdir():
                dest = local_dir / child.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(child), str(dest))
            shutil.rmtree(nested, ignore_errors=True)
        return
    except Exception:
        pass

    # Fallback: walk + get_file_contents
    names = await _listdir(service, remote_dir)
    for name in names:
        remote = f"{remote_dir.rstrip('/')}/{name}"
        local = local_dir / name
        if await _isdir(service, remote):
            await _pull_remote_dir(service, remote, local)
        else:
            await _pull_remote_file(service, remote, local)


async def _push_bytes(service, data: bytes, remote_path: str) -> None:
    parent = str(Path(remote_path).parent).replace("\\", "/")
    if parent and parent not in (".", "/"):
        await _makedirs(service, parent)
    await _afc(service.set_file_contents(remote_path, data))


async def _push_file(service, local: Path, remote_path: str) -> None:
    await _push_bytes(service, local.read_bytes(), remote_path)


async def find_remote_kits(service) -> list[tuple[str, str]]:
    """
    Return list of (kit_name, remote_kit_dir) under Documents/Backups.
    Supports Backups/<date>/<kit>/ and Backups/<kit>/.
    """
    found: list[tuple[str, str]] = []
    backups = _docs_join("Backups")
    if not await _exists(service, backups):
        return found
    for name in await _listdir(service, backups):
        path = f"{backups}/{name}"
        if not await _isdir(service, path):
            continue
        manifest = f"{path}/manifest.json"
        if await _exists(service, manifest):
            found.append((name, path))
            continue
        for child in await _listdir(service, path):
            cpath = f"{path}/{child}"
            if await _exists(service, f"{cpath}/manifest.json"):
                found.append((child, cpath))
    return found


async def pull_kits(bundle_id: str) -> dict:
    ensure_library()
    seed_library_from_backups()
    result = {
        "pulled": [],
        "skipped": [],
        "errors": [],
        "library": str(LIBRARY_DIR),
    }
    lockdown, service = await _open_docs(bundle_id)
    try:
        remote_kits = await find_remote_kits(service)
        for kit_name, remote_dir in remote_kits:
            dest = LIBRARY_DIR / kit_name
            try:
                # Cheap check: pull manifest only first
                remote_hash = None
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_man = Path(tmp) / "manifest.json"
                    try:
                        await _pull_remote_file(
                            service, f"{remote_dir}/manifest.json", tmp_man
                        )
                        remote_hash = _kit_audio_hash(Path(tmp))
                    except Exception:
                        remote_hash = None

                    local_hash = _kit_audio_hash(dest) if dest.exists() else None
                    if dest.exists() and remote_hash and remote_hash == local_hash:
                        # Same audio — optionally fill missing local tags from phone
                        local_tags = dest / "tags.json"
                        if not local_tags.exists():
                            try:
                                await _pull_remote_file(
                                    service,
                                    f"{remote_dir}/tags.json",
                                    local_tags,
                                )
                                result["pulled"].append(
                                    {"kit": kit_name, "action": "tags-from-phone"}
                                )
                            except Exception:
                                result["skipped"].append(kit_name)
                        else:
                            result["skipped"].append(kit_name)
                        continue

                    # Full kit copy (new or changed audio)
                    tmp_path = Path(tmp) / kit_name
                    await _pull_remote_dir(service, remote_dir, tmp_path)
                    if not (tmp_path / "manifest.json").exists():
                        matches = list(tmp_path.rglob("manifest.json"))
                        if matches:
                            tmp_path = matches[0].parent
                    if not (tmp_path / "manifest.json").exists():
                        result["errors"].append(f"No manifest in pulled {kit_name}")
                        continue
                    if dest.exists():
                        # Preserve newer Mac tags.json across audio refresh
                        mac_tags = dest / "tags.json"
                        saved_tags = None
                        if mac_tags.exists():
                            saved_tags = mac_tags.read_bytes()
                        shutil.rmtree(dest)
                        shutil.copytree(tmp_path, dest)
                        if saved_tags is not None:
                            (dest / "tags.json").write_bytes(saved_tags)
                    else:
                        shutil.copytree(tmp_path, dest)
                    result["pulled"].append({"kit": kit_name, "action": "copied"})
            except Exception as e:
                result["errors"].append(f"{kit_name}: {e}")
        return result
    finally:
        await service.close()


async def push_tags(bundle_id: str, force: bool = False) -> dict:
    ensure_library()
    result = {
        "pushed": [],
        "skipped": [],
        "errors": [],
    }
    kits = list_local_kits()
    if not kits:
        return result

    lockdown, service = await _open_docs(bundle_id)
    try:
        import_root = _docs_join("Import")
        await _makedirs(service, import_root)

        state = load_sync_state()
        kit_state = state.setdefault("kits", {})

        for kit in kits:
            tags_path = kit / "tags.json"
            man_path = kit / "manifest.json"
            if not tags_path.exists() or not man_path.exists():
                result["skipped"].append({"kit": kit.name, "reason": "missing tags/manifest"})
                continue
            fp = _tags_fingerprint(tags_path)
            prev = kit_state.get(kit.name, {})
            if (
                not force
                and prev.get("tagsFingerprint") == fp
                and prev.get("pushed")
            ):
                result["skipped"].append({"kit": kit.name, "reason": "unchanged"})
                continue

            remote_kit = f"{import_root}/{kit.name}"
            await _makedirs(service, remote_kit)
            try:
                await _push_file(service, man_path, f"{remote_kit}/manifest.json")
                await _push_file(service, tags_path, f"{remote_kit}/tags.json")
                # Empty annotations stub helps some importers
                anns = kit / "annotations.json"
                if anns.exists():
                    await _push_file(service, anns, f"{remote_kit}/annotations.json")
                else:
                    await _push_bytes(
                        service,
                        b'{"annotations":[]}\n',
                        f"{remote_kit}/annotations.json",
                    )
                try:
                    tags = _load_json(tags_path)
                    n = len(tags.get("tags", tags if isinstance(tags, list) else []))
                except Exception:
                    n = 0
                kit_state[kit.name] = {
                    "tagsFingerprint": fp,
                    "pushed": True,
                    "pushedAt": _now_iso(),
                    "tagCount": n,
                    "audioContentHash": _kit_audio_hash(kit),
                }
                result["pushed"].append({"kit": kit.name, "tagCount": n})
            except Exception as e:
                result["errors"].append(f"{kit.name}: {e}")

        state["lastSyncAt"] = _now_iso()
        save_sync_state(state)
        return result
    finally:
        await service.close()


async def full_sync(bundle_id: str, force: bool = False) -> dict:
    status = await discover_device(bundle_id)
    if not status.get("connected"):
        return {
            "ok": False,
            "status": status,
            "pull": None,
            "push": None,
            "error": status.get("error") or "iPhone not connected",
        }
    pull = await pull_kits(bundle_id)
    push = await push_tags(bundle_id, force=force)
    return {"ok": True, "status": status, "pull": pull, "push": push, "error": None}


def sync_to_dict(result: dict) -> dict:
    """JSON-serializable summary for the review UI."""
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="BabyTalk USB sync (Mac ↔ iPhone)")
    parser.add_argument(
        "command",
        choices=["status", "sync", "pull", "push", "seed"],
        help="status | sync (pull+push) | pull | push | seed",
    )
    parser.add_argument(
        "--bundle-id",
        default=DEFAULT_BUNDLE_ID,
        help=f"App bundle id (default {DEFAULT_BUNDLE_ID})",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-push tags even if fingerprint unchanged",
    )
    args = parser.parse_args(argv[1:])

    ensure_library()

    if args.command == "seed":
        n = seed_library_from_backups()
        msg = {"seeded": n, "library": str(LIBRARY_DIR)}
        print(json.dumps(msg, indent=2) if args.json else f"Seeded {n} kit(s) into {LIBRARY_DIR}")
        return 0

    async def run():
        if args.command == "status":
            return await discover_device(args.bundle_id)
        if args.command == "pull":
            status = await discover_device(args.bundle_id)
            if not status.get("connected"):
                return {"ok": False, "status": status, "error": status.get("error")}
            pull = await pull_kits(args.bundle_id)
            return {"ok": True, "status": status, "pull": pull}
        if args.command == "push":
            status = await discover_device(args.bundle_id)
            if not status.get("connected"):
                return {"ok": False, "status": status, "error": status.get("error")}
            push = await push_tags(args.bundle_id, force=args.force)
            return {"ok": True, "status": status, "push": push}
        return await full_sync(args.bundle_id, force=args.force)

    try:
        result = asyncio.run(run())
    except Exception as e:
        result = {"ok": False, "error": str(e)}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if args.command == "status":
            if result.get("connected"):
                print(
                    f"Connected: {result.get('deviceName') or 'iPhone'} · "
                    f"Backups={result.get('hasBackups')} Import={result.get('hasImport')}"
                )
            else:
                print(f"Not connected: {result.get('error')}")
        else:
            if not result.get("ok", True) and result.get("error"):
                print(f"Error: {result['error']}")
                return 1
            st = result.get("status") or {}
            print(f"Device: {st.get('deviceName') or 'iPhone'}")
            if result.get("pull"):
                p = result["pull"]
                print(
                    f"Pull: {len(p.get('pulled', []))} updated, "
                    f"{len(p.get('skipped', []))} skipped, "
                    f"{len(p.get('errors', []))} errors"
                )
                for e in p.get("errors", [])[:5]:
                    print(f"  ! {e}")
            if result.get("push"):
                p = result["push"]
                print(
                    f"Push: {len(p.get('pushed', []))} kits, "
                    f"{len(p.get('skipped', []))} skipped, "
                    f"{len(p.get('errors', []))} errors"
                )
                for item in p.get("pushed", []):
                    print(f"  → {item['kit']} ({item.get('tagCount', 0)} tags)")
                for e in p.get("errors", [])[:5]:
                    print(f"  ! {e}")
            print(f"Library: {LIBRARY_DIR}")
            print("Open the BabyTalk app on the phone so tags auto-import.")
    return 0 if result.get("ok", True) or args.command == "status" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
