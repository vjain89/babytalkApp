"""Local Whisper suggestions for BabyTalk snippet spans (optional, never ground truth).

Uses faster-whisper offline. Human word/phonetic fields stay authoritative.

Requires:
  tools/.venv/bin/pip install faster-whisper

Usage:
  python3 tools/asr_suggest.py /path/to/kit --start-ms 1200 --end-ms 2100
  python3 tools/asr_suggest.py /path/to/kit --uuid <annotation-uuid>
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install deps: pip install numpy soundfile\n" + str(e)
    ) from e

# Default small/fast model; override with BABYTALK_WHISPER_MODEL (e.g. tiny, small, medium).
DEFAULT_MODEL = os.environ.get("BABYTALK_WHISPER_MODEL", "base")
SOURCE = "whisper_local"
PAD_MS = 100
MIN_SLICE_MS = 80

# Map Review Browser language → Whisper language code (best-effort).
LANGUAGE_HINTS = {
    "Swiss German dialect": "de",
    "Spanish": "es",
    "English": "en",
}

_model = None
_model_name: str | None = None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def get_model(model_size: str = DEFAULT_MODEL):
    """Lazy-load and cache one Whisper model in-process."""
    global _model, _model_name
    if _model is not None and _model_name == model_size:
        return _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Install faster-whisper: tools/.venv/bin/pip install faster-whisper"
        ) from e
    # CPU int8 is the practical default on Apple Silicon without extra torch GPU setup.
    _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _model_name = model_size
    return _model


def resolve_audio(kit: Path) -> Path:
    manifest = load_json(kit / "manifest.json")
    name = manifest.get("audioFile", "audio.wav")
    path = kit / name
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    return path


def slice_audio(
    audio_path: Path,
    start_ms: int,
    end_ms: int,
    pad_ms: int = PAD_MS,
) -> tuple[np.ndarray, int]:
    """Return mono float32 samples and sample rate for [start,end] with pad."""
    data, sr = sf.read(str(audio_path), always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    n = len(data)
    start = max(0, int((start_ms - pad_ms) * sr / 1000))
    end = min(n, int((end_ms + pad_ms) * sr / 1000))
    if end <= start:
        raise ValueError("Empty audio slice")
    chunk = np.asarray(data[start:end], dtype=np.float32)
    return chunk, int(sr)


def write_wav_temp(samples: np.ndarray, sr: int) -> Path:
    """Write 16-bit mono WAV for Whisper (temp file)."""
    # faster-whisper can take ndarray+sr directly; still normalize to float32 path.
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 1.0:
        samples = samples / peak
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    path = Path(name)
    # soundfile write is fine; Whisper accepts the path.
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return path


def transcribe_slice(
    audio_path: Path,
    start_ms: int,
    end_ms: int,
    *,
    language_hint: str | None = None,
    model_size: str = DEFAULT_MODEL,
) -> dict:
    """Run local Whisper on a time slice. Returns suggestion dict (not ground truth)."""
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    if end_ms - start_ms < MIN_SLICE_MS:
        end_ms = start_ms + MIN_SLICE_MS

    samples, sr = slice_audio(audio_path, start_ms, end_ms)
    model = get_model(model_size)

    whisper_lang = None
    if language_hint:
        whisper_lang = LANGUAGE_HINTS.get(language_hint.strip()) or language_hint.strip()
        if len(whisper_lang) > 3:
            whisper_lang = LANGUAGE_HINTS.get(language_hint)  # unknown → auto

    tmp: Path | None = None
    try:
        tmp = write_wav_temp(samples, sr)
        segments, info = model.transcribe(
            str(tmp),
            language=whisper_lang,
            beam_size=1,
            vad_filter=False,
        )
        texts = [s.text.strip() for s in segments if (s.text or "").strip()]
        text = " ".join(texts).strip()
        detected = getattr(info, "language", None) or whisper_lang
        return {
            "text": text,
            "language": detected,
            "model": f"faster-whisper/{model_size}",
            "source": SOURCE,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "startMs": start_ms,
            "endMs": end_ms,
        }
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def find_item_by_uuid(kit: Path, uuid: str) -> tuple[str, dict]:
    """Return ('annotation'|'tag', item) for uuid."""
    if (kit / "annotations.json").exists():
        anns = load_json(kit / "annotations.json")
        items = anns.get("annotations", anns if isinstance(anns, list) else [])
        for a in items:
            if a.get("uuid") == uuid:
                return "annotation", a
    if (kit / "tags.json").exists():
        tags = load_json(kit / "tags.json")
        items = tags.get("tags", tags if isinstance(tags, list) else [])
        for t in items:
            if t.get("uuid") == uuid:
                return "tag", t
    raise KeyError(f"No annotation/tag with uuid {uuid}")


def suggest_for_kit_uuid(
    kit: Path,
    uuid: str,
    *,
    language_hint: str | None = None,
    model_size: str = DEFAULT_MODEL,
    persist: bool = True,
) -> dict:
    """Transcribe one annotation/tag span and optionally write asr back to JSON."""
    kind, item = find_item_by_uuid(kit, uuid)
    start_ms = int(item.get("startMs") if item.get("startMs") is not None else item.get("tMs") or 0)
    end_ms = item.get("endMs")
    if end_ms is None:
        end_ms = start_ms + 300
    end_ms = int(end_ms)
    hint = language_hint or item.get("language")
    audio = resolve_audio(kit)
    asr = transcribe_slice(
        audio,
        start_ms,
        end_ms,
        language_hint=hint,
        model_size=model_size,
    )
    item["asr"] = asr
    if persist:
        if kind == "annotation":
            path = kit / "annotations.json"
            data = load_json(path)
            items = data.get("annotations", data if isinstance(data, list) else [])
            for i, a in enumerate(items):
                if a.get("uuid") == uuid:
                    items[i] = item
                    break
            write_json(path, {"annotations": items})
        else:
            path = kit / "tags.json"
            data = load_json(path)
            items = data.get("tags", data if isinstance(data, list) else [])
            for i, t in enumerate(items):
                if t.get("uuid") == uuid:
                    items[i] = item
                    break
            write_json(path, {"tags": items})
    return {"ok": True, "kind": kind, "uuid": uuid, "asr": asr, "item": item}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local Whisper suggestion for a kit span")
    p.add_argument("kit", type=Path, help="Path to session kit folder")
    p.add_argument("--uuid", help="Annotation or tag uuid")
    p.add_argument("--start-ms", type=int)
    p.add_argument("--end-ms", type=int)
    p.add_argument("--language", help="Optional language hint (Swiss German dialect / Spanish / English / de|es|en)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-persist", action="store_true")
    args = p.parse_args(argv)

    kit = args.kit.expanduser().resolve()
    if not kit.is_dir():
        print(f"Not a kit folder: {kit}")
        return 1

    if args.uuid:
        result = suggest_for_kit_uuid(
            kit,
            args.uuid,
            language_hint=args.language,
            model_size=args.model,
            persist=not args.no_persist,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.start_ms is None or args.end_ms is None:
        print("Provide --uuid or both --start-ms and --end-ms")
        return 1
    audio = resolve_audio(kit)
    asr = transcribe_slice(
        audio,
        args.start_ms,
        args.end_ms,
        language_hint=args.language,
        model_size=args.model,
    )
    print(json.dumps({"ok": True, "asr": asr}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
