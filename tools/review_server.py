"""BabyTalk Mac review studio (P3).

Local browser UI over the Mac BabyTalk library (or an explicit kit/backup path):
  - play audio
  - drag on the waveform to select a span
  - add / edit / delete tags with category + optional speaker
  - verbal: word (required) + optional phonetic
  - non-verbal vocalization: optional phonetic (+ optional note); vegetative: optional note
  - find speech segments (VAD) → provisional ML candidates
  - confirm or dismiss ML candidates (assign category + speaker / word+phonetic)
  - Sync with iPhone (USB) to pull kits and push tags.json

Usage:
  python3 tools/review_server.py
  # or: python3 tools/review_server.py ~/Documents/BabyTalk/Library
  open http://127.0.0.1:8765

USB sync needs tools/.venv (pymobiledevice3). See tools/README.md.
Speech VAD needs: pip install numpy soundfile
"""

from __future__ import annotations

import json
import mimetypes
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from babytalk_paths import (  # noqa: E402
    DEFAULT_BUNDLE_ID,
    LIBRARY_DIR,
    ensure_library,
    seed_library_from_backups,
)

ROOT: Path = Path(".")
BUNDLE_ID = DEFAULT_BUNDLE_ID
VENV_PYTHON = TOOLS_DIR / ".venv" / "bin" / "python"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def list_kits(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    )


def resolve_kit(name: str) -> Path:
    if name and (ROOT / name).exists():
        return ROOT / name
    if (ROOT / "manifest.json").exists():
        return ROOT
    raise FileNotFoundError(name)


def kit_payload(kit: Path) -> dict:
    manifest = load_json(kit / "manifest.json")
    tags = []
    if (kit / "tags.json").exists():
        tp = load_json(kit / "tags.json")
        tags = tp.get("tags", tp if isinstance(tp, list) else [])
    anns = []
    if (kit / "annotations.json").exists():
        ap = load_json(kit / "annotations.json")
        anns = ap.get("annotations", ap if isinstance(ap, list) else [])
    folder = kit.name if kit != ROOT else kit.name
    return {
        "folder": folder,
        "manifest": manifest,
        "tags": tags,
        "annotations": anns,
        "audioUrl": f"/audio?kit={folder}",
        "tagsPath": str((kit / "tags.json").resolve()),
    }


def read_tags(kit: Path) -> list:
    path = kit / "tags.json"
    if not path.exists():
        return []
    tp = load_json(path)
    return tp.get("tags", tp if isinstance(tp, list) else [])


def write_tags(kit: Path, tags: list) -> None:
    write_json(kit / "tags.json", {"tags": tags})


def read_annotations(kit: Path) -> list:
    path = kit / "annotations.json"
    if not path.exists():
        return []
    ap = load_json(path)
    return ap.get("annotations", ap if isinstance(ap, list) else [])


def write_annotations(kit: Path, anns: list) -> None:
    write_json(kit / "annotations.json", {"annotations": anns})


# Primary Review Browser taxonomy (human assigns on confirm; VAD does not).
CATEGORIES = (
    "verbal vocalization",
    "non-verbal vocalization",
    "non-vocal vegetative sound",
)
SPEAKER_PRESETS = ("Baby", "Parent", "Other")


def normalize_category(value) -> str:
    text = (value or "").strip()
    if text in CATEGORIES:
        return text
    return ""


def normalize_speaker(value) -> str:
    return (value or "").strip()


def compose_label(
    category: str = "",
    speaker: str = "",
    word: str = "",
    note: str = "",
) -> str:
    """Keep `label` a useful string for older UI / phone import.

    Format: ``category · speaker · detail`` where detail is the intended
    ``word`` (verbal) or free-form ``note`` (non-verbal / vegetative).
    Phonetic is never folded into label — it lives in its own field.
    """
    category = (category or "").strip()
    speaker = (speaker or "").strip()
    detail = (word or "").strip() or (note or "").strip()
    parts = [p for p in (category, speaker, detail) if p]
    return " · ".join(parts) if parts else "untitled"


def apply_taxonomy_fields(item: dict, body: dict) -> None:
    """Set category / speaker / word / phonetic / note / label from a request body."""
    has_tax = any(
        k in body for k in ("category", "speaker", "note", "word", "phonetic")
    )
    if has_tax:
        category = normalize_category(body.get("category"))
        speaker = normalize_speaker(body.get("speaker"))
        word = str(body.get("word") or "").strip()
        phonetic = str(body.get("phonetic") or "").strip()
        note = str(body.get("note") or "").strip()
        if category:
            item["category"] = category
        else:
            item.pop("category", None)
        if speaker:
            item["speaker"] = speaker
        else:
            item.pop("speaker", None)

        if category == "verbal vocalization":
            if word:
                item["word"] = word
            else:
                item.pop("word", None)
            if phonetic:
                item["phonetic"] = phonetic
            else:
                item.pop("phonetic", None)
            item.pop("note", None)
            item["label"] = compose_label(category, speaker, word=word)
        elif category == "non-verbal vocalization":
            item.pop("word", None)
            if phonetic:
                item["phonetic"] = phonetic
            else:
                item.pop("phonetic", None)
            if note:
                item["note"] = note
            else:
                item.pop("note", None)
            # Phonetic stays in its own field; label uses optional note only.
            item["label"] = compose_label(category, speaker, note=note)
        else:
            item.pop("word", None)
            item.pop("phonetic", None)
            if note:
                item["note"] = note
            else:
                item.pop("note", None)
            item["label"] = compose_label(category, speaker, note=note)
        return

    # Legacy clients that only send label.
    if "label" in body:
        item["label"] = (body.get("label") or item.get("label") or "").strip() or "untitled"


def taxonomy_validation_error(body: dict) -> str | None:
    """Return an error string if taxonomy fields are invalid, else None."""
    category = normalize_category(body.get("category"))
    if not category:
        return "category required"
    if category == "verbal vocalization" and not str(body.get("word") or "").strip():
        return "word required for verbal vocalization"
    return None


def run_vad_for_kit(kit: Path, body: dict | None = None) -> dict:
    """Run energy-based VAD and merge results into annotations.json."""
    body = body or {}
    try:
        from vad_segments import (
            MERGE_GAP_MS,
            MIN_DUR_MS,
            SPEECH_DELTA_DB,
            process_kit,
        )
    except ImportError as e:
        return {
            "ok": False,
            "error": "VAD deps missing. Install with: pip install numpy soundfile",
            "detail": str(e),
        }

    def _num(key: str, default: float) -> float:
        val = body.get(key)
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    try:
        result = process_kit(
            kit,
            merge_gap_ms=_num("mergeGapMs", MERGE_GAP_MS),
            min_dur_ms=_num("minDurMs", MIN_DUR_MS),
            speech_delta_db=_num("speechDeltaDb", SPEECH_DELTA_DB),
            write=True,
        )
    except Exception as e:  # noqa: BLE001 — surface to UI
        return {"ok": False, "error": str(e), "kit": kit.name}
    return result


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BabyTalk Review</title>
<style>
  :root {
    --bg: #f3efe8;
    --ink: #1a1a1a;
    --muted: #667;
    --panel: #fffdf9;
    --line: #ddd5c8;
    --accent: #2f6fed;
    --tag: #c45c26;
    --ml: #5a7a5a;
    --sel: rgba(47, 111, 237, 0.28);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    background: var(--bg);
    color: var(--ink);
  }
  header {
    padding: 14px 20px 12px;
    background: var(--ink);
    color: #f7f3ec;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  header .top-row {
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
  }
  header strong { font-size: 18px; letter-spacing: 0.02em; }
  header .hint { color: #bdb7ae; font-size: 13px; font-family: ui-sans-serif, system-ui, sans-serif; flex: 1; }
  .sync-btn {
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 13px;
    padding: 7px 12px;
    border-radius: 6px;
    border: 1px solid #5a5a5a;
    background: #2a2a2a;
    color: #f7f3ec;
    cursor: pointer;
  }
  .sync-btn:hover { background: #3a3a3a; }
  .sync-btn:disabled { opacity: 0.55; cursor: wait; }
  #vocabBar {
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 13px;
    color: #e8e0d4;
  }
  #vocabBar .vocab-stats { color: #cfc6b8; }
  #vocabBar .vocab-stats b { color: #fff; font-weight: 600; }
  #saveStatus {
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 12px;
    color: #9dceb0;
    min-height: 1.2em;
  }
  #saveStatus.error { color: #f0a0a0; }
  #saveStatus.idle { color: #8a847a; }
  main { display: grid; grid-template-columns: 260px 1fr; min-height: calc(100vh - 56px); }
  #sidebar {
    border-right: 1px solid var(--line);
    background: #ebe4da;
    padding: 12px;
    overflow: auto;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  #sidebar h3 {
    margin: 4px 0 10px; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--muted);
  }
  #kitList button {
    display: block; width: 100%; text-align: left; margin: 0 0 6px;
    padding: 10px; border: 1px solid var(--line); background: var(--panel);
    cursor: pointer; border-radius: 6px; font: inherit;
  }
  #kitList button.active { border-color: var(--ink); background: #fff; box-shadow: 0 1px 0 rgba(0,0,0,0.06); }
  #kitList .year-group { margin: 0 0 14px; }
  #kitList .year-label {
    font-size: 13px; font-weight: 700; color: var(--ink);
    margin: 0 0 6px; letter-spacing: 0.02em;
  }
  #kitList .month-label {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); margin: 10px 0 6px;
  }
  #sessionTitle {
    cursor: text; border-radius: 6px; padding: 2px 6px; margin-left: -6px;
    outline: none;
  }
  #sessionTitle:hover { background: rgba(0,0,0,0.04); }
  #sessionTitle.editing {
    background: #fff; border: 1px solid var(--line);
    box-shadow: 0 0 0 2px rgba(47,111,237,0.15);
  }
  #sessionTitleInput {
    font: inherit; font-size: 1.5rem; font-weight: 600;
    width: min(100%, 420px); padding: 4px 8px;
    border: 1px solid var(--line); border-radius: 6px;
  }
  .title-row {
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    margin: 0 0 4px;
  }
  .title-row button.ghost {
    border: 1px solid var(--line); background: #fff; padding: 5px 10px;
    border-radius: 6px; cursor: pointer; font: inherit; font-size: 12px;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  #panel { padding: 18px 22px 40px; max-width: 1100px; }
  #cloudPane {
    margin: 18px 0 8px;
  }
  #cloudPane h3 {
    margin: 0 0 6px; font-size: 16px;
  }
  #cloudMeta {
    font-size: 13px; color: var(--muted);
    font-family: ui-sans-serif, system-ui, sans-serif;
    margin: 0 0 8px;
  }
  #wordCloud {
    display: block;
    width: 100%;
    height: 200px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    cursor: default;
  }
  #wordCloud.has-words { cursor: pointer; }
  .muted { color: var(--muted); font-size: 13px; font-family: ui-sans-serif, system-ui, sans-serif; }
  .sans { font-family: ui-sans-serif, system-ui, sans-serif; }
  h2 { margin: 0 0 4px; font-weight: 600; }
  .transport {
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    margin: 12px 0; font-family: ui-sans-serif, system-ui, sans-serif;
  }
  .transport button, .tag-form button, .row button {
    border: 1px solid var(--line); background: #fff; padding: 7px 12px;
    border-radius: 6px; cursor: pointer; font: inherit;
  }
  .transport button.primary, .tag-form button.primary {
    background: var(--ink); color: #fff; border-color: var(--ink);
  }
  #waveWrap {
    position: relative;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px 8px 0 0;
    overflow: hidden;
    user-select: none;
    touch-action: none;
  }
  #wave { display: block; width: 100%; height: 160px; cursor: crosshair; }
  #overview {
    position: relative;
    height: 28px;
    border: 1px solid var(--line);
    border-top: none;
    border-radius: 0 0 8px 8px;
    background: #ebe4da;
    cursor: grab;
    overflow: hidden;
  }
  #overviewCanvas { display: block; width: 100%; height: 28px; }
  #overviewWindow {
    position: absolute; top: 0; bottom: 0;
    background: rgba(47, 111, 237, 0.22);
    border: 1px solid var(--accent);
    pointer-events: none;
  }
  #playhead {
    position: absolute; top: 0; bottom: 0; width: 2px; background: #c62828;
    pointer-events: none; left: 0; z-index: 4;
  }
  #tagMarks {
    position: absolute; inset: 0;
    pointer-events: none;
    z-index: 1;
  }
  .tag-mark {
    position: absolute; top: 0; bottom: 0;
    background: rgba(194, 100, 48, 0.20);
    border-left: 2px solid #c26430;
    border-right: 1px solid rgba(194, 100, 48, 0.55);
    box-sizing: border-box;
    overflow: hidden;
  }
  .tag-mark.point {
    border-right: none;
    width: 2px !important;
    background: #c26430;
  }
  .tag-mark span {
    position: absolute; top: 3px; left: 4px; right: 2px;
    font: 600 10px/1.2 ui-sans-serif, system-ui, sans-serif;
    color: #7a3410;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    background: rgba(255, 253, 249, 0.82);
    padding: 1px 4px;
    border-radius: 2px;
    max-width: 100%;
    box-sizing: border-box;
  }
  #selection {
    position: absolute; top: 0; bottom: 0; background: var(--sel);
    pointer-events: none; display: none;
    z-index: 2;
  }
  .sel-handle {
    position: absolute; top: 0; bottom: 0; width: 12px;
    margin-left: -6px; cursor: ew-resize;
    pointer-events: auto; z-index: 3;
    background: linear-gradient(90deg, transparent 0 3px, var(--accent) 3px 9px, transparent 9px 12px);
  }
  #selHandleL { left: 0; }
  #selHandleR { left: 100%; }
  .zoom-bar {
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    margin: 8px 0 0;
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
  }
  .zoom-bar #zoomLabel { color: var(--muted); min-width: 140px; }
  .tag-form {
    display: grid;
    grid-template-columns: 1fr 110px 110px auto;
    gap: 8px;
    margin: 14px 0 8px;
    font-family: ui-sans-serif, system-ui, sans-serif;
    align-items: start;
  }
  .tag-form input, .tag-form select {
    padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; font: inherit;
    background: #fff; color: var(--ink);
  }
  .tax-block {
    display: grid;
    gap: 6px;
  }
  .speaker-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }
  .speaker-row input {
    flex: 1 1 120px;
    min-width: 100px;
    padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; font: inherit;
  }
  .chip {
    border: 1px solid var(--line);
    background: #fff;
    color: var(--ink);
    padding: 5px 10px;
    border-radius: 999px;
    cursor: pointer;
    font: inherit;
    font-size: 12px;
  }
  .chip.active {
    background: var(--ink);
    color: #fff;
    border-color: var(--ink);
  }
  .note-input {
    padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; font: inherit;
  }
  .detail-fields {
    display: grid;
    gap: 6px;
  }
  .detail-fields[data-mode="verbal"] .note-field { display: none; }
  .detail-fields[data-mode="nonverbal"] .word-field { display: none; }
  .detail-fields[data-mode="note"] .word-field,
  .detail-fields[data-mode="note"] .phonetic-field { display: none; }
  .detail-fields[data-mode=""] .word-field,
  .detail-fields[data-mode=""] .phonetic-field,
  .detail-fields[data-mode=""] .note-field { display: none; }
  .help {
    background: #e7f0ff; border: 1px solid #c5d7f5; border-radius: 8px;
    padding: 10px 12px; margin: 0 0 14px;
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
  }
  .section { margin-top: 22px; }
  .section h3 { margin: 0 0 8px; font-size: 16px; }
  .meta-form {
    display: grid;
    gap: 12px;
    font-family: ui-sans-serif, system-ui, sans-serif;
    max-width: 640px;
  }
  .meta-form label {
    display: grid;
    gap: 4px;
    font-size: 13px;
    color: var(--muted);
  }
  .meta-form label span.field-name {
    color: var(--ink);
    font-weight: 600;
    font-size: 13px;
  }
  .meta-form input,
  .meta-form textarea {
    padding: 8px 10px;
    border: 1px solid var(--line);
    border-radius: 6px;
    font: inherit;
    font-size: 14px;
    color: var(--ink);
    background: #fff;
    width: 100%;
    box-sizing: border-box;
  }
  .meta-form textarea {
    min-height: 88px;
    resize: vertical;
    line-height: 1.4;
  }
  .meta-actions {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .meta-actions button.primary {
    border: 1px solid var(--ink);
    background: var(--ink);
    color: #fff;
    padding: 8px 14px;
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
  }
  .row {
    display: grid;
    grid-template-columns: 150px 1fr;
    gap: 10px; align-items: start;
    padding: 10px 0; border-bottom: 1px solid var(--line);
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 14px;
  }
  .row-fields {
    display: grid;
    gap: 8px;
  }
  .row-fields .controls {
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  }
  .row select, .row input[type=text], .row .note-input {
    padding: 6px 8px; border: 1px solid var(--line); border-radius: 4px;
    font: inherit; background: #fff; color: var(--ink);
  }
  .row .category-select { width: 100%; max-width: 280px; }
  .row .speaker-row input { padding: 6px 8px; }
  .pill {
    font-size: 11px; padding: 3px 7px; border-radius: 4px; background: #efe6da;
    white-space: normal; line-height: 1.35;
  }
  .pill.user { background: #f3d9c8; color: #7a3410; }
  .pill.ml { background: #d9e6d9; color: #2f4f2f; }
  .pill .sub { display: block; color: inherit; opacity: 0.85; font-weight: 500; }
  audio { display: none; }
  @media (max-width: 800px) {
    main { grid-template-columns: 1fr; }
    .tag-form { grid-template-columns: 1fr 1fr; }
    .row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <div class="top-row">
    <strong>BabyTalk</strong>
    <span class="hint">Mac review · tags write live to each kit’s <code style="color:#e8dcc8">tags.json</code></span>
    <button type="button" id="btnSync" class="sync-btn">Sync with iPhone</button>
  </div>
  <div id="vocabBar">
    <div class="vocab-stats" id="vocabStats">Loading vocabulary…</div>
  </div>
  <div id="saveStatus" class="idle">Tags save immediately to disk when you Add / Save / Delete.</div>
</header>
<main>
  <aside id="sidebar">
    <h3>Sessions</h3>
    <div id="kitList"></div>
  </aside>
  <section id="panel"><p class="muted">Select a session kit.</p></section>
</main>
<audio id="audio" preload="auto"></audio>
<script>
let kits = [];
let current = null;
let audioBuf = null;
let durationSec = 0;
let selStart = null; // seconds (lo after normalize)
let selEnd = null;   // seconds (hi after normalize)
let dragging = false;
let dragAnchor = null;
let handleDrag = null; // 'left' | 'right' | null
let panning = false;
let panAnchorX = 0;
let panAnchorViewStart = 0;
/** Visible window on the main waveform (seconds). */
let viewStart = 0;
let viewDur = 1;
const MIN_VIEW_DUR = 0.05; // 50ms
const MIN_SNIPPET = 0.02; // 20ms
let followPlayhead = true;
/** Authoritative playhead in seconds (Web Audio playback). */
let playheadSec = 0;
let audioCtx = null;
let activeSource = null;
let rafId = null;
let playbackOriginCtx = 0;
let playbackOriginBuf = 0;
let playbackEndBuf = null;
/** Where to park the cursor when a scheduled clip ends (snippet start, not file 0). */
let playbackParkBuf = 0;

const audioEl = () => document.getElementById('audio');

const CATEGORIES = [
  'verbal vocalization',
  'non-verbal vocalization',
  'non-vocal vegetative sound',
];
const SPEAKER_PRESETS = ['Baby', 'Parent', 'Other'];

function categoryOptionsHtml(selected) {
  const sel = selected || '';
  let html = `<option value="">Category…</option>`;
  for (const c of CATEGORIES) {
    html += `<option value="${esc(c)}"${c === sel ? ' selected' : ''}>${esc(c)}</option>`;
  }
  return html;
}

function speakerChipsHtml(selected, inputId) {
  const sel = (selected || '').trim();
  const chips = SPEAKER_PRESETS.map(s =>
    `<button type="button" class="chip${s === sel ? ' active' : ''}" data-speaker="${esc(s)}">${esc(s)}</button>`
  ).join('');
  return `<div class="speaker-row">
    ${chips}
    <input id="${esc(inputId)}" type="text" value="${esc(sel)}" placeholder="Speaker (optional)" autocomplete="off"/>
  </div>`;
}

function wireSpeakerChips(root, inputEl) {
  if (!root || !inputEl) return;
  const sync = () => {
    const v = inputEl.value.trim();
    root.querySelectorAll('.chip[data-speaker]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.speaker === v);
    });
  };
  root.querySelectorAll('.chip[data-speaker]').forEach(btn => {
    btn.onclick = () => {
      const name = btn.dataset.speaker || '';
      inputEl.value = inputEl.value.trim() === name ? '' : name;
      sync();
    };
  });
  inputEl.addEventListener('input', sync);
  sync();
}

function freeformNote(item) {
  // Prefer explicit persisted fields when present.
  if ((item.note || '').trim()) return String(item.note).trim();
  if ((item.word || '').trim()) return String(item.word).trim();
  const label = (item.label || '').trim();
  if (!label) return '';
  const cat = (item.category || '').trim();
  const sp = (item.speaker || '').trim();
  const catSp = [cat, sp].filter(Boolean).join(' · ');
  if (catSp && label === catSp) return '';
  if (CATEGORIES.includes(label)) return '';
  if (catSp && label.startsWith(catSp + ' · ')) return label.slice(catSp.length + 3);
  if (cat && label.startsWith(cat + ' · ')) return label.slice(cat.length + 3);
  const composed = cat && sp ? `${cat} · ${sp}` : (cat || sp);
  if (composed && label === composed) return '';
  return label;
}

function itemWord(item) {
  if ((item.word || '').trim()) return String(item.word).trim();
  if ((item.category || '') === 'verbal vocalization') return freeformNote(item);
  return '';
}

function itemPhonetic(item) {
  return (item.phonetic || '').trim();
}

function itemNote(item) {
  if ((item.note || '').trim()) return String(item.note).trim();
  const cat = (item.category || '');
  if (cat === 'verbal vocalization') return '';
  // Don't treat phonetic as a free-form note for non-verbal.
  if (cat === 'non-verbal vocalization' && (item.phonetic || '').trim()) return '';
  return freeformNote(item);
}

function detailModeForCategory(cat) {
  if (cat === 'verbal vocalization') return 'verbal';
  if (cat === 'non-verbal vocalization') return 'nonverbal';
  if (cat) return 'note';
  return '';
}

function detailSummaryHtml(item) {
  const bits = [];
  if ((item.word || '').trim()) {
    const w = String(item.word).trim();
    const ph = (item.phonetic || '').trim();
    bits.push(ph ? `${w} · ${ph}` : w);
  } else if ((item.phonetic || '').trim()) {
    bits.push(String(item.phonetic).trim());
  }
  if ((item.note || '').trim()) bits.push(String(item.note).trim());
  return bits.map(b => `<span class="sub">${esc(b)}</span>`).join('');
}

function detailFieldsHtml(item, idPrefix) {
  const cat = (item && item.category) || '';
  const mode = detailModeForCategory(cat);
  const word = itemWord(item || {});
  const phonetic = itemPhonetic(item || {});
  const note = itemNote(item || {});
  const notePh = cat === 'non-verbal vocalization'
    ? 'Optional note (e.g. squeal, rasp)'
    : 'Optional note (e.g. sneeze, cough)';
  return `<div class="detail-fields" data-mode="${esc(mode)}" data-detail-root="${esc(idPrefix)}">
    <div class="word-field">
      <input class="note-input" type="text" data-field="word" value="${esc(word)}" placeholder="Word (required) — e.g. Lorenzo" autocomplete="off"/>
    </div>
    <div class="phonetic-field">
      <input class="note-input" type="text" data-field="phonetic" value="${esc(phonetic)}" placeholder="Phonetic (optional) — e.g. na nen zo" autocomplete="off"/>
    </div>
    <div class="note-field">
      <input class="note-input" type="text" data-field="note" value="${esc(note)}" placeholder="${esc(notePh)}" autocomplete="off"/>
    </div>
  </div>`;
}

function syncDetailFields(scope) {
  if (!scope) return;
  const cat = (scope.querySelector('[data-field="category"], .category-select, #categoryInput')?.value || '').trim();
  const root = scope.querySelector('.detail-fields');
  if (!root) return;
  root.dataset.mode = detailModeForCategory(cat);
}

function wireCategoryDetail(scope) {
  if (!scope) return;
  const sel = scope.querySelector('[data-field="category"], .category-select, #categoryInput');
  if (!sel) return;
  sel.addEventListener('change', () => syncDetailFields(scope));
  syncDetailFields(scope);
}

function readTaxonomyFrom(scope) {
  const category = (scope.querySelector('[data-field="category"], .category-select, #categoryInput')?.value || '').trim();
  const speaker = (scope.querySelector('[data-field="speaker"]')?.value || '').trim();
  const word = (scope.querySelector('[data-field="word"]')?.value || '').trim();
  const phonetic = (scope.querySelector('[data-field="phonetic"]')?.value || '').trim();
  const note = (scope.querySelector('[data-field="note"]')?.value || '').trim();
  return { category, speaker, word, phonetic, note };
}

function validateTaxonomy(tax) {
  if (!tax.category) return 'Pick a category';
  if (tax.category === 'verbal vocalization' && !tax.word) {
    return 'Enter the word (what he’s trying to say) for verbal vocalization';
  }
  return null;
}

function taxonomyPayload(tax) {
  const payload = {
    category: tax.category,
    speaker: tax.speaker,
  };
  if (tax.category === 'verbal vocalization') {
    payload.word = tax.word;
    payload.phonetic = tax.phonetic;
  } else if (tax.category === 'non-verbal vocalization') {
    payload.phonetic = tax.phonetic;
    payload.note = tax.note;
  } else {
    payload.note = tax.note;
  }
  return payload;
}
function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

function isBufferPlaying() {
  return activeSource != null;
}

function stopBufferPlayback() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (activeSource) {
    try { activeSource.onended = null; activeSource.stop(); } catch (e) {}
    activeSource = null;
  }
  playbackEndBuf = null;
}

function finishBufferPlayback(parkAt) {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (activeSource) {
    try { activeSource.onended = null; } catch (e) {}
    // Don't call stop() here — source already ended, or we're about to replace it.
    activeSource = null;
  }
  playbackEndBuf = null;
  const park = Number.isFinite(parkAt) ? parkAt : playbackParkBuf;
  playheadSec = Math.max(0, Math.min(durationSec || 0, park));
  updatePlayButton();
  paintOverlays();
  syncClock();
}

function startBufferPlayback(fromSec, toSec) {
  if (!audioBuf) return false;
  stopBufferPlayback();
  const ctx = getAudioCtx();
  if (ctx.state === 'suspended') ctx.resume();

  const start = Math.max(0, Math.min(fromSec, durationSec - 0.001));
  let end = toSec == null ? durationSec : toSec;
  end = Math.max(start + 0.001, Math.min(end, durationSec));
  const dur = end - start;

  const source = ctx.createBufferSource();
  source.buffer = audioBuf;
  source.connect(ctx.destination);
  playbackOriginCtx = ctx.currentTime;
  playbackOriginBuf = start;
  playbackEndBuf = end;
  // Always park back at the clip start (snippet lo), never file 0 unless that is the start.
  playbackParkBuf = start;
  playheadSec = start;
  activeSource = source;

  source.onended = () => {
    if (activeSource !== source) return;
    finishBufferPlayback(playbackParkBuf);
  };

  source.start(0, start, dur);
  tickPlayhead();
  return true;
}

function tickPlayhead() {
  if (!activeSource) return;
  const ctx = getAudioCtx();
  playheadSec = playbackOriginBuf + (ctx.currentTime - playbackOriginCtx);

  if (playbackEndBuf != null && playheadSec >= playbackEndBuf - 0.001) {
    // Stop the node so onended does not also fire with a stale park.
    const park = playbackParkBuf;
    try {
      activeSource.onended = null;
      activeSource.stop();
    } catch (e) {}
    activeSource = null;
    finishBufferPlayback(park);
    return;
  }

  if (followPlayhead && viewDur < durationSec - 1e-6) {
    const margin = viewDur * 0.15;
    if (playheadSec > viewEnd() - margin || playheadSec < viewStart + margin) {
      viewStart = playheadSec - viewDur * 0.35;
      clampView();
      updateZoomLabel();
      drawWave();
      drawOverview();
    }
  }
  syncClock();
  updatePlayButton();
  paintOverlays();
  rafId = requestAnimationFrame(tickPlayhead);
}

function syncClock() {
  const clock = document.getElementById('clock');
  if (clock) clock.textContent = `${playheadSec.toFixed(1)}s / ${durationSec.toFixed(1)}s`;
}

function setPlayhead(t) {
  playheadSec = Math.max(0, Math.min(durationSec || 0, t));
  // Keep hidden <audio> in sync for any leftover readers.
  const a = audioEl();
  if (a && a.readyState >= 1) {
    try { a.currentTime = playheadSec; } catch (e) {}
  }
  syncClock();
  paintOverlays();
}

function viewEnd() { return Math.min(durationSec, viewStart + viewDur); }

function resetView() {
  viewStart = 0;
  viewDur = Math.max(MIN_VIEW_DUR, durationSec || 1);
}

function clampView() {
  if (!durationSec) return;
  viewDur = Math.min(Math.max(MIN_VIEW_DUR, viewDur), durationSec);
  viewStart = Math.min(Math.max(0, viewStart), Math.max(0, durationSec - viewDur));
}

function zoomAt(factor, centerSec) {
  if (!durationSec) return;
  const c = centerSec == null ? viewStart + viewDur / 2 : centerSec;
  const rel = viewDur > 0 ? (c - viewStart) / viewDur : 0.5;
  viewDur = Math.min(durationSec, Math.max(MIN_VIEW_DUR, viewDur * factor));
  viewStart = c - rel * viewDur;
  clampView();
  updateZoomLabel();
  drawWave();
  drawOverview();
  paintOverlays();
}

function zoomBy(factor) {
  zoomAt(factor, viewStart + viewDur / 2);
}

function fitAll() {
  resetView();
  updateZoomLabel();
  drawWave();
  drawOverview();
  paintOverlays();
}

function zoomToSelection() {
  if (!hasSnippet() || !durationSec) return;
  const a = selStart, b = selEnd;
  const pad = Math.max(0.05, (b - a) * 0.15);
  viewStart = Math.max(0, a - pad);
  viewDur = Math.min(durationSec - viewStart, Math.max(MIN_VIEW_DUR, b - a + 2 * pad));
  clampView();
  updateZoomLabel();
  drawWave();
  drawOverview();
  paintOverlays();
}

function hasSnippet() {
  return selStart != null && selEnd != null && (selEnd - selStart) >= MIN_SNIPPET;
}

function normalizeSel() {
  if (selStart == null || selEnd == null) return;
  if (selStart > selEnd) {
    const t = selStart; selStart = selEnd; selEnd = t;
  }
}

function pauseIfPlaying() {
  if (isBufferPlaying()) {
    const ctx = getAudioCtx();
    playheadSec = playbackOriginBuf + (ctx.currentTime - playbackOriginCtx);
    if (playbackEndBuf != null) {
      playheadSec = Math.min(playheadSec, playbackEndBuf);
    }
    stopBufferPlayback();
    updatePlayButton();
    paintOverlays();
    syncClock();
    return;
  }
  const a = audioEl();
  if (a && !a.paused) a.pause();
}

async function playToggle() {
  if (isBufferPlaying()) {
    pauseIfPlaying();
    return;
  }
  try {
    if (audioBuf) {
      if (hasSnippet()) {
        normalizeSel();
        startBufferPlayback(selStart, selEnd);
      } else {
        startBufferPlayback(playheadSec, null);
      }
      updatePlayButton();
      paintOverlays();
      return;
    }
    // Fallback: HTML audio if decode failed
    const a = audioEl();
    if (!a.paused) {
      a.pause();
      updatePlayButton();
      return;
    }
    if (hasSnippet()) {
      normalizeSel();
      a.currentTime = selStart;
      await new Promise((r) => setTimeout(r, 50));
    }
    await a.play();
  } catch (err) {
    console.warn('play failed', err);
  }
  updatePlayButton();
  paintOverlays();
}

function updatePlayButton() {
  const btn = document.getElementById('btnPlay');
  if (!btn) return;
  const playing = isBufferPlaying() || (audioEl() && !audioEl().paused);
  if (playing && hasSnippet()) btn.textContent = 'Playing snippet';
  else btn.textContent = playing ? 'Pause' : 'Play';
}

function updateZoomLabel() {
  const el = document.getElementById('zoomLabel');
  if (!el || !durationSec) return;
  const x = durationSec / viewDur;
  el.textContent = x <= 1.01
    ? `Zoom 1× · full ${durationSec.toFixed(1)}s`
    : `Zoom ${x.toFixed(1)}× · ${viewStart.toFixed(2)}–${viewEnd().toFixed(2)}s`;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/** Prefer explicit `word` when present; else tokenize label. */
function wordsFromTags(tags) {
  const counts = new Map();
  for (const t of tags || []) {
    const intended = (t.word || '').trim().toLowerCase();
    if (intended) {
      counts.set(intended, (counts.get(intended) || 0) + 1);
      continue;
    }
    const parts = String(t.label || '').toLowerCase().match(/[a-z0-9']+/g) || [];
    for (const w of parts) {
      if (!w) continue;
      counts.set(w, (counts.get(w) || 0) + 1);
    }
  }
  return counts;
}

let cloudHits = []; // {word, count, x0,y0,x1,y1} in canvas CSS pixels

function updateVocabBar() {
  const statsEl = document.getElementById('vocabStats');
  if (!statsEl) return;
  const allTags = (kits || []).flatMap(k => k.tags || []);
  const allWords = wordsFromTags(allTags);
  statsEl.innerHTML =
    `<b>${allWords.size}</b> unique words · <b>${allTags.length}</b> tags across all kits`;
  renderWordCloud();
}

function renderWordCloud() {
  const canvas = document.getElementById('wordCloud');
  const meta = document.getElementById('cloudMeta');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 1000;
  const cssH = 200;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  canvas.style.height = cssH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fffdf9';
  ctx.fillRect(0, 0, cssW, cssH);
  cloudHits = [];

  if (!current) {
    if (meta) meta.textContent = 'Select a session';
    canvas.classList.remove('has-words');
    ctx.fillStyle = '#999';
    ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
    ctx.fillText('Word sizes reflect tag frequency', 14, cssH / 2);
    return;
  }

  const tags = current.tags || [];
  const counts = wordsFromTags(tags);
  const entries = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  if (meta) {
    meta.textContent = entries.length
      ? `${entries.length} unique · ${tags.length} tags in this kit · click a word to seek`
      : 'No tags yet in this kit';
  }
  if (!entries.length) {
    canvas.classList.remove('has-words');
    ctx.fillStyle = '#999';
    ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
    ctx.fillText('Add tags to build a word cloud', 14, cssH / 2);
    return;
  }
  canvas.classList.add('has-words');

  const maxC = entries[0][1];
  const minC = entries[entries.length - 1][1];
  const sizeFor = (n) => {
    if (maxC === minC) return 20;
    return 11 + ((n - minC) / (maxC - minC)) * 26;
  };
  const colors = ['#7a3410', '#c26430', '#1a1a1a', '#5a3a28', '#8b4513', '#2f4f2f'];
  const placed = [];
  const pad = 3;
  const overlaps = (a, b) => !(a.x1 + pad < b.x0 || a.x0 - pad > b.x1 || a.y1 + pad < b.y0 || a.y0 - pad > b.y1);
  const cx = cssW / 2;
  const cy = cssH / 2;

  for (let i = 0; i < entries.length; i++) {
    const [word, n] = entries[i];
    const size = sizeFor(n);
    ctx.font = `600 ${size}px "Iowan Old Style", Palatino, Georgia, serif`;
    const tw = ctx.measureText(word).width;
    const th = size * 1.05;
    let box = null;
    let angle = i * 0.7;
    let radius = 0;
    for (let step = 0; step < 2500; step++) {
      const x = cx + Math.cos(angle) * radius - tw / 2;
      const y = cy + Math.sin(angle) * radius;
      const cand = { x0: x, y0: y - th * 0.78, x1: x + tw, y1: y + th * 0.28 };
      const inBounds = cand.x0 >= 4 && cand.y0 >= 4 && cand.x1 <= cssW - 4 && cand.y1 <= cssH - 4;
      if (inBounds && !placed.some(p => overlaps(cand, p))) {
        box = cand;
        break;
      }
      angle += 0.28;
      radius += 0.45;
    }
    if (!box) continue;
    placed.push(box);
    ctx.fillStyle = colors[i % colors.length];
    ctx.globalAlpha = 0.55 + 0.45 * ((n - minC) / Math.max(1, maxC - minC));
    ctx.font = `600 ${size}px "Iowan Old Style", Palatino, Georgia, serif`;
    ctx.fillText(word, box.x0, box.y0 + th * 0.78);
    ctx.globalAlpha = 1;
    cloudHits.push({ word, count: n, ...box });
  }
}

function seekWordFromCloud(word) {
  if (!current || !word) return;
  const tags = current.tags || [];
  const re = new RegExp(`(?:^|[^a-z0-9'])${word.replace(/'/g, "\\'")}(?:$|[^a-z0-9'])`, 'i');
  const tag = tags.find(t => re.test(String(t.label || '').toLowerCase()))
    || tags.find(t => String(t.label || '').toLowerCase().includes(word));
  if (!tag) return;
  pauseIfPlaying();
  const a = (tag.startMs || 0) / 1000;
  const b = tag.endMs != null ? tag.endMs / 1000 : a + 0.3;
  selStart = a; selEnd = b; normalizeSel();
  setPlayhead(selStart);
  syncSelInputs();
  paintOverlays();
  startBufferPlayback(selStart, selEnd);
}

function wireWordCloud() {
  const canvas = document.getElementById('wordCloud');
  if (!canvas) return;
  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const hit = cloudHits.find(h => x >= h.x0 && x <= h.x1 && y >= h.y0 && y <= h.y1);
    if (hit) seekWordFromCloud(hit.word);
  };
}

function flashSaveStatus(message, ok = true) {
  const el = document.getElementById('saveStatus');
  if (!el) return;
  el.className = ok ? '' : 'error';
  el.textContent = message;
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const text = await res.text();
  let data = null;
  try { data = JSON.parse(text); } catch (e) {}
  if (!res.ok) {
    throw new Error((data && data.error) || text || res.statusText);
  }
  return data || { ok: true };
}

function kitCreatedAt(k) {
  const t = k && k.manifest && k.manifest.createdAt;
  const n = Number(t);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

/** YY_MM_DD__HH:MM:SS from createdAt (local time). */
function stampNameFromMs(ms) {
  const d = new Date(ms || Date.now());
  const yy = String(d.getFullYear()).slice(-2);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${yy}_${mm}_${dd}__${hh}:${mi}:${ss}`;
}

function displayName(k) {
  return (k.manifest && (k.manifest.sessionName || k.manifest.originalSessionName)) || k.folder;
}

function originalName(k) {
  const m = k.manifest || {};
  return m.originalSessionName || null;
}

function setActiveKit(i) {
  document.querySelectorAll('#kitList button').forEach((b) => {
    b.classList.toggle('active', Number(b.dataset.i) === i);
  });
}

async function renameCurrentKit(newName) {
  const name = String(newName || '').trim();
  if (!current || !name) return;
  const data = await postJson('/api/kit/rename', {
    kit: current.folder,
    sessionName: name,
  });
  if (data.manifest) current.manifest = data.manifest;
  flashSaveStatus(`Renamed to “${name}”`);
  await softRefreshCurrent();
  const title = document.getElementById('sessionTitle');
  if (title) title.textContent = name;
  const orig = document.getElementById('originalNameLine');
  if (orig) {
    const o = originalName(current);
    orig.textContent = o && o !== name ? `Original: ${o}` : '';
  }
}

function wireSessionTitle() {
  const title = document.getElementById('sessionTitle');
  const btnStamp = document.getElementById('btnStampName');
  if (!title || !current) return;

  const startEdit = () => {
    if (document.getElementById('sessionTitleInput')) return;
    const input = document.createElement('input');
    input.id = 'sessionTitleInput';
    input.type = 'text';
    input.value = displayName(current);
    title.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const prev = displayName(current);
      const val = String(input.value || '').trim();
      const h2 = document.createElement('h2');
      h2.id = 'sessionTitle';
      h2.title = 'Click to rename';
      h2.textContent = save && val ? val : prev;
      if (input.parentNode) input.replaceWith(h2);
      wireSessionTitle();
      if (save && val && val !== prev) {
        try { await renameCurrentKit(val); }
        catch (err) { flashSaveStatus('Rename failed: ' + err.message, false); }
      }
    };
    input.onkeydown = (e) => {
      if (e.key === 'Enter') { e.preventDefault(); void finish(true); }
      if (e.key === 'Escape') { e.preventDefault(); void finish(false); }
    };
    input.onblur = () => { void finish(true); };
  };

  title.onclick = startEdit;
  if (btnStamp) {
    btnStamp.onclick = async () => {
      const stamp = stampNameFromMs(kitCreatedAt(current));
      try { await renameCurrentKit(stamp); }
      catch (err) { flashSaveStatus('Rename failed: ' + err.message, false); }
    };
  }
}

function readMetaFormValues() {
  return {
    voiceNotesOriginalFilename: (document.getElementById('metaVoiceNotes')?.value || '').trim(),
    recordingLocation: (document.getElementById('metaLocation')?.value || '').trim(),
    contextNotes: (document.getElementById('metaContext')?.value || '').trim(),
  };
}

async function saveMetaForm() {
  if (!current) return;
  const fields = readMetaFormValues();
  const data = await postJson('/api/kit/metadata', {
    kit: current.folder,
    ...fields,
  });
  if (data.manifest) current.manifest = data.manifest;
  const hint = document.getElementById('metaSaveHint');
  if (hint) hint.textContent = 'Saved';
  flashSaveStatus('Session notes saved');
  await softRefreshCurrent();
  // softRefresh does not rebuild the panel — keep form values as typed.
  if (hint) {
    setTimeout(() => { if (hint.textContent === 'Saved') hint.textContent = ''; }, 2000);
  }
}

function wireMetaForm() {
  const btn = document.getElementById('btnSaveMeta');
  if (!btn || !current) return;
  btn.onclick = async () => {
    try { await saveMetaForm(); }
    catch (err) { flashSaveStatus('Notes save failed: ' + err.message, false); }
  };
}

async function refresh() {
  const res = await fetch('/api/kits');
  kits = await res.json();
  const el = document.getElementById('kitList');
  if (!el) return;

  // Group by year → month (local), newest first.
  const groups = new Map(); // year -> Map(monthIndex -> kits[])
  kits.forEach((k, i) => {
    k._i = i;
    const d = new Date(kitCreatedAt(k) || Date.now());
    const y = d.getFullYear();
    const m = d.getMonth();
    if (!groups.has(y)) groups.set(y, new Map());
    const months = groups.get(y);
    if (!months.has(m)) months.set(m, []);
    months.get(m).push(k);
  });
  const years = [...groups.keys()].sort((a, b) => b - a);
  const monthNames = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December',
  ];

  let html = '';
  for (const y of years) {
    html += `<div class="year-group"><div class="year-label">${y}</div>`;
    const months = groups.get(y);
    const monthIdxs = [...months.keys()].sort((a, b) => b - a);
    for (const mi of monthIdxs) {
      html += `<div class="month-label">${monthNames[mi]}</div>`;
      const list = months.get(mi).slice().sort((a, b) => kitCreatedAt(b) - kitCreatedAt(a));
      for (const k of list) {
        const nTags = (k.tags || []).length;
        const nOpen = (k.annotations || []).filter(a => a.status !== 'confirmed' && a.status !== 'dismissed').length;
        const name = displayName(k);
        const orig = originalName(k);
        const origBit = orig && orig !== name
          ? `<br/><span class="muted">was ${esc(orig)}</span>`
          : '';
        html += `<button data-i="${k._i}">${esc(name)}${origBit}<br/>
          <span class="muted">${((k.manifest.durationMs||0)/1000).toFixed(1)}s · ${nTags} tags · ${nOpen} candidates</span></button>`;
      }
    }
    html += `</div>`;
  }
  el.innerHTML = html || '<p class="muted">No kits found.</p>';
  el.querySelectorAll('button').forEach(b => b.onclick = () => select(+b.dataset.i));
  updateVocabBar();
}

/** Reload kit metadata/lists without resetting zoom, playhead, or audio. */
async function softRefreshCurrent() {
  const folder = current && current.folder;
  if (!folder) return;
  await refresh();
  const i = kits.findIndex(x => x.folder === folder);
  if (i < 0) return;
  current = kits[i];
  setActiveKit(i);
  renderLists();
  drawOverview();
  paintOverlays();
  updateVocabBar();
}

async function select(i) {
  current = kits[i];
  setActiveKit(i);
  selStart = null; selEnd = null;
  audioBuf = null;
  resetView();
  renderShell();
  await loadAudioAndWave();
  updateVocabBar();
}

function renderShell() {
  const k = current;
  if (!k) return;
  document.getElementById('panel').innerHTML = `
    <div class="help">
      <strong>How to tag:</strong> click to set playhead · drag to select a snippet · pick a <strong>category</strong> · optional <strong>speaker</strong>.
      For <strong>verbal vocalization</strong>, enter the <strong>word</strong> (required) and optional <strong>phonetic</strong>;
      for <strong>non-verbal vocalization</strong>, optional <strong>phonetic</strong> (+ optional note);
      vegetative sounds take an optional note · Add tag.
      Existing tags show as orange bands on the waveform (and overview strip).
      <strong>Play:</strong> with no selection, plays from the playhead; with a selection, loops that snippet only.
      Drag the blue handles to adjust snippet ends (pauses if playing). Scroll to zoom · Shift-drag to pan.
      Space plays/pauses. Use <strong>Sync with iPhone</strong> (USB) to pull kits and push tags — open the app on the phone so tags auto-import.
    </div>
    <div class="title-row">
      <h2 id="sessionTitle" title="Click to rename">${esc(displayName(k))}</h2>
      <button type="button" class="ghost" id="btnStampName" title="Set name to YY_MM_DD__HH:MM:SS from recording time">Use date stamp</button>
    </div>
    <p class="muted sans" id="originalNameLine">${(() => {
      const o = originalName(k);
      const n = displayName(k);
      return o && o !== n ? `Original: ${esc(o)}` : '';
    })()}</p>
    <p class="muted sans">${esc(k.manifest.recordingUuid)} · hash ${esc(String(k.manifest.audioContentHash||'').slice(0,12))}…</p>
    <p class="muted sans" id="tagsPathLine">Saving to <code>${esc(k.tagsPath || (k.folder + '/tags.json'))}</code></p>
    <div class="transport sans">
      <button class="primary" id="btnPlay">Play</button>
      <button id="btnMarkIn">Mark in</button>
      <button id="btnMarkOut">Mark out</button>
      <button id="btnClearSel">Clear selection</button>
      <span class="muted" id="clock">0.0s / 0.0s</span>
    </div>
    <div id="waveWrap">
      <canvas id="wave" width="1000" height="160"></canvas>
      <div id="tagMarks"></div>
      <div id="selection">
        <div id="selHandleL" class="sel-handle" title="Drag start"></div>
        <div id="selHandleR" class="sel-handle" title="Drag end"></div>
      </div>
      <div id="playhead"></div>
    </div>
    <div id="overview" title="Drag to pan · click to jump">
      <canvas id="overviewCanvas" width="1000" height="28"></canvas>
      <div id="overviewWindow"></div>
    </div>
    <div class="zoom-bar">
      <button id="btnZoomIn" title="Zoom in">Zoom +</button>
      <button id="btnZoomOut" title="Zoom out">Zoom −</button>
      <button id="btnFit">Fit all</button>
      <button id="btnZoomSel">Zoom to selection</button>
      <label class="muted"><input type="checkbox" id="chkFollow" checked/> Follow playhead</label>
      <span id="zoomLabel" class="muted">Zoom 1×</span>
    </div>
    <div class="tag-form">
      <div class="tax-block">
        <select id="categoryInput" class="category-select" data-field="category" title="Category">
          <option value="">Category…</option>
          <option value="verbal vocalization">verbal vocalization</option>
          <option value="non-verbal vocalization">non-verbal vocalization</option>
          <option value="non-vocal vegetative sound">non-vocal vegetative sound</option>
        </select>
        <div class="speaker-row" id="addSpeakerRow">
          <button type="button" class="chip" data-speaker="Baby">Baby</button>
          <button type="button" class="chip" data-speaker="Parent">Parent</button>
          <button type="button" class="chip" data-speaker="Other">Other</button>
          <input id="speakerInput" data-field="speaker" type="text" placeholder="Speaker (optional)" autocomplete="off"/>
        </div>
        <div class="detail-fields" id="addDetailFields" data-mode="">
          <div class="word-field">
            <input id="wordInput" class="note-input" data-field="word" type="text" placeholder="Word (required) — e.g. Lorenzo" autocomplete="off"/>
          </div>
          <div class="phonetic-field">
            <input id="phoneticInput" class="note-input" data-field="phonetic" type="text" placeholder="Phonetic (optional) — e.g. na nen zo" autocomplete="off"/>
          </div>
          <div class="note-field">
            <input id="noteInput" class="note-input" data-field="note" type="text" placeholder="Optional note (e.g. sneeze, cough)"/>
          </div>
        </div>
      </div>
      <input id="startInput" type="number" step="0.01" min="0" placeholder="Start s"/>
      <input id="endInput" type="number" step="0.01" min="0" placeholder="End s"/>
      <button class="primary" id="btnAdd">Add tag</button>
    </div>
    <p class="muted" id="selHint">No selection — drag on the waveform or use Mark in / Mark out.</p>
    <div id="cloudPane" class="section">
      <h3>Word cloud</h3>
      <div id="cloudMeta" class="muted">Building from this kit’s tags…</div>
      <canvas id="wordCloud" width="1000" height="200"></canvas>
    </div>
    <div class="section">
      <h3>Tags</h3>
      <div id="tagList"></div>
    </div>
    <div class="section">
      <h3>ML candidates</h3>
      <p class="muted sans" style="margin:0 0 8px">
        Find speech-like spans with local VAD (merge gaps ≤0.4s, drop &lt;0.3s).
        VAD does not know who spoke — on confirm, assign a <strong>category</strong> and optional <strong>speaker</strong>
        (verbal: <strong>word</strong> + optional <strong>phonetic</strong>; non-verbal: optional <strong>phonetic</strong>).
        Confirm promotes into tags; dismiss hides. Re-run replaces provisional VAD/ml_v0 suggestions only.
      </p>
      <div class="meta-actions" style="margin:0 0 10px">
        <button type="button" class="primary" id="btnVad">Find speech segments</button>
        <span class="muted" id="vadHint"></span>
      </div>
      <div id="annList"></div>
    </div>
    <div class="section" id="metaSection">
      <h3>Session notes</h3>
      <p class="muted sans" style="margin:0 0 10px">Optional context for this recording. Saved into <code>manifest.json</code>.</p>
      <div class="meta-form">
        <label>
          <span class="field-name">Voice Notes / original filename</span>
          <input id="metaVoiceNotes" type="text" placeholder="e.g. New Recording 3" value="${esc(k.manifest.voiceNotesOriginalFilename || k.manifest.originalSessionName || '')}"/>
        </label>
        <label>
          <span class="field-name">Location</span>
          <input id="metaLocation" type="text" placeholder="e.g. kitchen, park, car" value="${esc(k.manifest.recordingLocation || '')}"/>
        </label>
        <label>
          <span class="field-name">Context</span>
          <textarea id="metaContext" placeholder="e.g. eating breakfast · reading before bedtime · walking around the neighborhood">${esc(k.manifest.contextNotes || '')}</textarea>
        </label>
        <div class="meta-actions">
          <button type="button" class="primary" id="btnSaveMeta">Save notes</button>
          <span class="muted" id="metaSaveHint"></span>
        </div>
      </div>
    </div>
  `;
  wireTransport();
  wireWordCloud();
  wireSessionTitle();
  wireMetaForm();
  wireVad();
  wireSpeakerChips(document.getElementById('addSpeakerRow'), document.getElementById('speakerInput'));
  wireCategoryDetail(document.querySelector('.tag-form'));
  renderLists();
  updateVocabBar();
}

function wireVad() {
  const btn = document.getElementById('btnVad');
  const hint = document.getElementById('vadHint');
  if (!btn) return;
  btn.onclick = async () => {
    if (!current) return;
    btn.disabled = true;
    btn.textContent = 'Finding…';
    if (hint) hint.textContent = 'Running local VAD…';
    try {
      const data = await postJson('/api/vad/run', { kit: current.folder });
      if (!data.ok) {
        flashSaveStatus('VAD failed: ' + (data.error || 'unknown'), false);
        if (hint) hint.textContent = data.error || 'failed';
        return;
      }
      await softRefreshCurrent();
      const n = data.added != null ? data.added : 0;
      flashSaveStatus(`Found ${n} speech segment(s) → annotations.json`);
      if (hint) hint.textContent = `${n} new · ${data.total || n} total on disk`;
    } catch (err) {
      flashSaveStatus('VAD failed: ' + err.message, false);
      if (hint) hint.textContent = err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Find speech segments';
    }
  };
}

function wireTransport() {
  document.getElementById('btnPlay').onclick = () => { void playToggle(); };
  document.getElementById('btnMarkIn').onclick = () => {
    pauseIfPlaying();
    selStart = playheadSec;
    if (selEnd == null || selEnd < selStart + MIN_SNIPPET) {
      selEnd = Math.min(durationSec, selStart + 0.3);
    }
    normalizeSel();
    syncSelInputs(); paintOverlays();
  };
  document.getElementById('btnMarkOut').onclick = () => {
    pauseIfPlaying();
    selEnd = playheadSec;
    if (selStart == null || selStart > selEnd - MIN_SNIPPET) {
      selStart = Math.max(0, selEnd - 0.3);
    }
    normalizeSel();
    syncSelInputs(); paintOverlays();
  };
  document.getElementById('btnClearSel').onclick = () => {
    pauseIfPlaying();
    selStart = selEnd = null; syncSelInputs(); paintOverlays();
  };
  document.getElementById('btnAdd').onclick = addTag;
  document.getElementById('btnZoomIn').onclick = () => zoomBy(0.7);
  document.getElementById('btnZoomOut').onclick = () => zoomBy(1 / 0.7);
  document.getElementById('btnFit').onclick = fitAll;
  document.getElementById('btnZoomSel').onclick = zoomToSelection;
  document.getElementById('chkFollow').onchange = (e) => { followPlayhead = e.target.checked; };
  document.getElementById('startInput').onchange = () => {
    const v = parseFloat(document.getElementById('startInput').value);
    if (Number.isNaN(v)) return;
    pauseIfPlaying();
    selStart = Math.max(0, Math.min(v, (selEnd ?? durationSec) - MIN_SNIPPET));
    if (selEnd == null) selEnd = Math.min(durationSec, selStart + 0.3);
    normalizeSel();
    syncSelInputs(); paintOverlays();
  };
  document.getElementById('endInput').onchange = () => {
    const v = parseFloat(document.getElementById('endInput').value);
    if (Number.isNaN(v)) return;
    pauseIfPlaying();
    selEnd = Math.min(durationSec, Math.max(v, (selStart ?? 0) + MIN_SNIPPET));
    if (selStart == null) selStart = Math.max(0, selEnd - 0.3);
    normalizeSel();
    syncSelInputs(); paintOverlays();
  };
  const canvas = document.getElementById('wave');
  canvas.onpointerdown = onPointerDown;
  canvas.onpointermove = onPointerMove;
  canvas.onpointerup = onPointerUp;
  canvas.onpointercancel = onPointerUp;
  canvas.addEventListener('wheel', onWaveWheel, { passive: false });
  wireHandles();
  wireOverview();
}

function wireHandles() {
  const left = document.getElementById('selHandleL');
  const right = document.getElementById('selHandleR');
  if (!left || !right) return;
  const startDrag = (which) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!hasSnippet()) return;
    pauseIfPlaying();
    handleDrag = which;
    left.setPointerCapture?.(e.pointerId);
    right.setPointerCapture?.(e.pointerId);
  };
  left.onpointerdown = startDrag('left');
  right.onpointerdown = startDrag('right');
  const move = (e) => {
    if (!handleDrag || !hasSnippet()) return;
    const t = xToTime(e.clientX);
    if (handleDrag === 'left') {
      selStart = Math.max(0, Math.min(t, selEnd - MIN_SNIPPET));
    } else {
      selEnd = Math.min(durationSec, Math.max(t, selStart + MIN_SNIPPET));
    }
    syncSelInputs();
    paintOverlays();
  };
  const up = () => { handleDrag = null; };
  left.onpointermove = move;
  right.onpointermove = move;
  left.onpointerup = up;
  right.onpointerup = up;
  left.onpointercancel = up;
  right.onpointercancel = up;
}

function syncSelInputs() {
  document.getElementById('startInput').value = selStart == null ? '' : selStart.toFixed(2);
  document.getElementById('endInput').value = selEnd == null ? '' : selEnd.toFixed(2);
  const hint = document.getElementById('selHint');
  if (!hasSnippet()) {
    hint.textContent = 'No selection — click to set playhead, or drag to select a snippet.';
  } else {
    hint.textContent = `Snippet ${selStart.toFixed(2)}s – ${selEnd.toFixed(2)}s (${((selEnd-selStart)*1000).toFixed(0)} ms) · Play loops this range`;
  }
}

function timeToX(t, width) {
  if (!viewDur) return 0;
  return ((t - viewStart) / viewDur) * width;
}

function paintTagMarks() {
  const layer = document.getElementById('tagMarks');
  const wrap = document.getElementById('waveWrap');
  if (!layer || !wrap || !durationSec) return;
  const w = wrap.clientWidth;
  const tags = (current && current.tags) || [];
  const ve = viewEnd();
  layer.innerHTML = tags.map(tag => {
    const a = (tag.startMs || 0) / 1000;
    const bRaw = tag.endMs != null ? tag.endMs / 1000 : a;
    const lo = Math.min(a, bRaw);
    const hi = Math.max(a, bRaw);
    const isPoint = (hi - lo) < 0.02;
    if (hi < viewStart || lo > ve) return '';
    const left = timeToX(Math.max(lo, viewStart), w);
    const right = timeToX(isPoint ? Math.min(lo + 0.001, ve) : Math.min(hi, ve), w);
    if (right < -2 || left > w + 2) return '';
    const width = Math.max(2, right - left);
    const label = esc(tag.label || '(untitled)');
    const tip = `${label} · ${lo.toFixed(2)}s${isPoint ? '' : ('–' + hi.toFixed(2) + 's')}`;
    const cls = isPoint ? 'tag-mark point' : 'tag-mark';
    const labelHtml = isPoint || width < 18 ? '' : `<span>${label}</span>`;
    return `<div class="${cls}" style="left:${left}px;width:${width}px" title="${tip}">${labelHtml}</div>`;
  }).join('');
}

function paintOverlays() {
  const wrap = document.getElementById('waveWrap');
  const box = document.getElementById('selection');
  const head = document.getElementById('playhead');
  if (!wrap || !durationSec) return;
  const w = wrap.clientWidth;
  const t = Number.isFinite(playheadSec) ? playheadSec : 0;

  paintTagMarks();

  if (head) {
    const x = timeToX(t, w);
    if (x < 0 || x > w) head.style.display = 'none';
    else {
      head.style.display = 'block';
      head.style.left = x + 'px';
    }
  }

  if (!box) return;
  if (!hasSnippet()) {
    box.style.display = 'none';
    paintOverviewWindow();
    return;
  }
  const a = selStart;
  const b = selEnd;
  const va = Math.max(a, viewStart);
  const vb = Math.min(b, viewEnd());
  if (vb <= va) {
    box.style.display = 'none';
    paintOverviewWindow();
    return;
  }
  box.style.display = 'block';
  box.style.left = timeToX(va, w) + 'px';
  box.style.width = Math.max(2, timeToX(vb, w) - timeToX(va, w)) + 'px';
  // Handles sit on true snippet ends when visible; hide when clipped off-screen.
  const hl = document.getElementById('selHandleL');
  const hr = document.getElementById('selHandleR');
  if (hl) hl.style.visibility = (a >= viewStart && a <= viewEnd()) ? 'visible' : 'hidden';
  if (hr) hr.style.visibility = (b >= viewStart && b <= viewEnd()) ? 'visible' : 'hidden';
  // Position handles at true ends relative to the (possibly clipped) box.
  if (hl) hl.style.left = ((a - va) / Math.max(vb - va, 1e-6)) * 100 + '%';
  if (hr) hr.style.left = ((b - va) / Math.max(vb - va, 1e-6)) * 100 + '%';
  paintOverviewWindow();
}

function xToTime(clientX) {
  const canvas = document.getElementById('wave');
  const rect = canvas.getBoundingClientRect();
  const x = Math.min(Math.max(0, clientX - rect.left), rect.width);
  return viewStart + (x / rect.width) * viewDur;
}

function onWaveWheel(e) {
  e.preventDefault();
  const center = xToTime(e.clientX);
  // trackpad pinch often sets ctrlKey; treat both scroll and pinch as zoom
  const dy = e.deltaY;
  const factor = dy > 0 ? 1.12 : 0.88;
  zoomAt(factor, center);
}

function onPointerDown(e) {
  if (!durationSec) return;
  if (handleDrag) return;
  // Shift / middle / alt = pan
  if (e.shiftKey || e.altKey || e.button === 1) {
    panning = true;
    panAnchorX = e.clientX;
    panAnchorViewStart = viewStart;
    e.target.setPointerCapture?.(e.pointerId);
    return;
  }
  pauseIfPlaying();
  dragging = true;
  dragAnchor = xToTime(e.clientX);
  selStart = dragAnchor;
  selEnd = dragAnchor;
  setPlayhead(dragAnchor);
  syncSelInputs(); paintOverlays();
  e.target.setPointerCapture?.(e.pointerId);
}

function onPointerMove(e) {
  if (handleDrag) return;
  if (panning) {
    const canvas = document.getElementById('wave');
    const w = canvas.getBoundingClientRect().width || 1;
    const dt = ((panAnchorX - e.clientX) / w) * viewDur;
    viewStart = panAnchorViewStart + dt;
    clampView();
    updateZoomLabel();
    drawWave();
    drawOverview();
    paintOverlays();
    return;
  }
  if (!dragging) return;
  selEnd = xToTime(e.clientX);
  syncSelInputs(); paintOverlays();
}

function onPointerUp(e) {
  if (handleDrag) return;
  if (panning) {
    panning = false;
    return;
  }
  if (!dragging) return;
  dragging = false;
  selEnd = xToTime(e.clientX);
  const clickThresh = Math.max(0.015, 0.02 * (viewDur / Math.max(durationSec, 0.001)));
  if (Math.abs(selEnd - selStart) < clickThresh) {
    // Click = set playhead only (no snippet). Next Play starts here.
    setPlayhead(dragAnchor);
    selStart = selEnd = null;
  } else {
    normalizeSel();
    setPlayhead(selStart);
  }
  syncSelInputs(); paintOverlays();
  updatePlayButton();
}

function wireOverview() {
  const ov = document.getElementById('overview');
  if (!ov) return;
  let draggingOv = false;
  ov.onpointerdown = (e) => {
    draggingOv = true;
    jumpOverview(e.clientX);
    ov.setPointerCapture?.(e.pointerId);
  };
  ov.onpointermove = (e) => { if (draggingOv) jumpOverview(e.clientX); };
  ov.onpointerup = () => { draggingOv = false; };
  ov.onpointercancel = () => { draggingOv = false; };
}

function jumpOverview(clientX) {
  const ov = document.getElementById('overview');
  const rect = ov.getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  const center = frac * durationSec;
  viewStart = center - viewDur / 2;
  clampView();
  updateZoomLabel();
  drawWave();
  drawOverview();
  paintOverlays();
}

function paintOverviewWindow() {
  const win = document.getElementById('overviewWindow');
  const ov = document.getElementById('overview');
  if (!win || !ov || !durationSec) return;
  const w = ov.clientWidth;
  win.style.left = (viewStart / durationSec) * w + 'px';
  win.style.width = Math.max(4, (viewDur / durationSec) * w) + 'px';
}

async function loadAudioAndWave() {
  stopBufferPlayback();
  playheadSec = 0;
  const a = audioEl();
  a.src = current.audioUrl + '&t=' + Date.now();
  // Warm the element (optional fallback path); don't rely on it for seeking.
  try { a.load(); } catch (e) {}
  try {
    const res = await fetch(current.audioUrl);
    const arr = await res.arrayBuffer();
    const ctx = getAudioCtx();
    if (ctx.state === 'suspended') await ctx.resume();
    audioBuf = await ctx.decodeAudioData(arr.slice(0));
    durationSec = audioBuf.duration || (current.manifest.durationMs || 0) / 1000;
    resetView();
    updateZoomLabel();
    drawWave();
    drawOverview();
    paintOverlays();
    syncClock();
  } catch (err) {
    durationSec = (current.manifest.durationMs || 0) / 1000;
    resetView();
    drawWaveEmpty();
    drawOverview();
    console.warn('Waveform decode failed', err);
  }
}

function drawWaveEmpty() {
  const canvas = document.getElementById('wave');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 1000;
  const h = 160;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fffdf9'; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#999'; ctx.font = '13px sans-serif';
  ctx.fillText('Waveform unavailable — you can still mark in/out while playing.', 16, h/2);
}

function drawChannelPeaks(ctx, data, sampleStart, sampleEnd, w, h) {
  const mid = h / 2;
  const span = Math.max(1, sampleEnd - sampleStart);
  ctx.strokeStyle = '#5c5c5c';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const s0 = sampleStart + Math.floor((x / w) * span);
    const s1 = sampleStart + Math.floor(((x + 1) / w) * span);
    let min = 1, max = -1;
    for (let i = s0; i < s1 && i < data.length; i++) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (s1 <= s0 && s0 < data.length) {
      min = max = data[s0];
    }
    ctx.moveTo(x, mid + min * mid * 0.92);
    ctx.lineTo(x, mid + max * mid * 0.92);
  }
  ctx.stroke();
  ctx.strokeStyle = '#ddd5c8';
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
}

function drawWave() {
  const canvas = document.getElementById('wave');
  if (!canvas || !audioBuf) return drawWaveEmpty();
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.parentElement.clientWidth || 1000;
  const h = 160;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fffdf9'; ctx.fillRect(0, 0, w, h);

  const data = audioBuf.getChannelData(0);
  const sr = audioBuf.sampleRate;
  const sampleStart = Math.floor(viewStart * sr);
  const sampleEnd = Math.min(data.length, Math.ceil(viewEnd() * sr));
  drawChannelPeaks(ctx, data, sampleStart, sampleEnd, w, h);

  // Time ticks
  ctx.fillStyle = '#888';
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
  const tickEvery = niceTick(viewDur);
  const first = Math.ceil(viewStart / tickEvery) * tickEvery;
  for (let t = first; t <= viewEnd() + 1e-9; t += tickEvery) {
    const x = timeToX(t, w);
    ctx.strokeStyle = '#ece6dc';
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillText(formatTick(t), x + 3, 12);
  }
  paintOverlays();
}

function niceTick(span) {
  const raw = span / 6;
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-4))));
  const n = raw / pow;
  let step = 1;
  if (n > 5) step = 10;
  else if (n > 2) step = 5;
  else if (n > 1) step = 2;
  return step * pow;
}

function formatTick(t) {
  if (viewDur < 2) return t.toFixed(2) + 's';
  if (viewDur < 30) return t.toFixed(1) + 's';
  return Math.floor(t) + 's';
}

function drawOverview() {
  const canvas = document.getElementById('overviewCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.parentElement.clientWidth || 1000;
  const h = 28;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#ebe4da'; ctx.fillRect(0, 0, w, h);
  if (audioBuf && durationSec) {
    const data = audioBuf.getChannelData(0);
    drawChannelPeaks(ctx, data, 0, data.length, w, h);
  }
  // Tag markers across the full timeline.
  if (durationSec && current && current.tags) {
    for (const tag of current.tags) {
      const a = (tag.startMs || 0) / 1000;
      const bRaw = tag.endMs != null ? tag.endMs / 1000 : a;
      const lo = Math.min(a, bRaw);
      const hi = Math.max(a, bRaw);
      const x0 = (lo / durationSec) * w;
      const x1 = (Math.max(hi, lo + 0.001) / durationSec) * w;
      ctx.fillStyle = 'rgba(194, 100, 48, 0.45)';
      ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    }
  }
  paintOverviewWindow();
}

function renderLists() {
  const k = current;
  const tags = k.tags || [];
  const tagList = document.getElementById('tagList');
  tagList.innerHTML = tags.length ? tags.map(t => `
    <div class="row" data-uuid="${esc(t.uuid)}">
      <span class="pill user">${((t.startMs||0)/1000).toFixed(2)}s${t.endMs!=null?('–'+(t.endMs/1000).toFixed(2)+'s'):''}
        ${t.category ? `<span class="sub">${esc(t.category)}</span>` : ''}
        ${t.speaker ? `<span class="sub">${esc(t.speaker)}</span>` : ''}
        ${detailSummaryHtml(t)}
      </span>
      <div class="row-fields">
        <select class="category-select" data-field="category">${categoryOptionsHtml(t.category || '')}</select>
        ${speakerChipsHtml(t.speaker || '', 'spk-' + t.uuid)}
        ${detailFieldsHtml(t, 'tag-' + t.uuid)}
        <div class="controls">
          <button data-act="seek">Seek</button>
          <button data-act="save">Save</button>
          <button data-act="delete">Delete</button>
        </div>
      </div>
    </div>`).join('') : '<p class="muted">No tags yet.</p>';

  tagList.querySelectorAll('.row').forEach(row => {
    const spInput = row.querySelector('input[id^="spk-"]');
    if (spInput) {
      spInput.dataset.field = 'speaker';
      wireSpeakerChips(row.querySelector('.speaker-row'), spInput);
    }
    wireCategoryDetail(row);
    row.querySelectorAll('button[data-act]').forEach(btn => btn.onclick = async () => {
      const uuid = row.dataset.uuid;
      const tag = tags.find(t => t.uuid === uuid);
      const act = btn.dataset.act;
      if (act === 'seek' && tag) {
        pauseIfPlaying();
        const a = (tag.startMs||0)/1000;
        const b = tag.endMs != null ? tag.endMs/1000 : a + 0.3;
        selStart = a; selEnd = b; normalizeSel();
        setPlayhead(selStart);
        syncSelInputs();
        startBufferPlayback(selStart, selEnd);
        return;
      }
      if (act === 'delete') {
        try {
          const data = await postJson('/api/tag/delete', { kit: k.folder, uuid });
          await softRefreshCurrent();
          flashSaveStatus(`Deleted · wrote ${data.tagsPath || (current && current.tagsPath) || 'tags.json'} · ${(current.tags||[]).length} tags on disk`);
        } catch (err) {
          flashSaveStatus('Delete failed: ' + err.message, false);
        }
        return;
      }
      if (act === 'save') {
        const tax = readTaxonomyFrom(row);
        const errMsg = validateTaxonomy(tax);
        if (errMsg) { alert(errMsg); return; }
        try {
          const data = await postJson('/api/tag/update', {
            kit: k.folder, uuid,
            ...taxonomyPayload(tax),
            startMs: tag.startMs, endMs: tag.endMs
          });
          await softRefreshCurrent();
          flashSaveStatus(`Updated · wrote ${data.tagsPath || current.tagsPath}`);
        } catch (err) {
          flashSaveStatus('Save failed: ' + err.message, false);
        }
      }
    });
  });

  const anns = (k.annotations||[]).filter(a => a.status !== 'dismissed');
  const annList = document.getElementById('annList');
  annList.innerHTML = anns.length ? anns.map(a => `
    <div class="row" data-uuid="${esc(a.uuid)}">
      <span class="pill ml">${((a.startMs||a.tMs||0)/1000).toFixed(2)}s${a.endMs!=null?('–'+(a.endMs/1000).toFixed(2)+'s'):''}${a.source?(' · '+esc(a.source)):''}${a.status==='confirmed'?' · confirmed':''}
        ${a.category ? `<span class="sub">${esc(a.category)}</span>` : ''}
        ${a.speaker ? `<span class="sub">${esc(a.speaker)}</span>` : ''}
        ${detailSummaryHtml(a)}
      </span>
      <div class="row-fields">
        <select class="category-select" data-field="category">${categoryOptionsHtml(a.category || '')}</select>
        ${speakerChipsHtml(a.speaker || '', 'ann-spk-' + a.uuid)}
        ${detailFieldsHtml(a, 'ann-' + a.uuid)}
        <div class="controls">
          <button data-act="seek">Seek</button>
          <button data-act="confirm">Confirm</button>
          <button data-act="dismiss">Dismiss</button>
        </div>
      </div>
    </div>`).join('') : '<p class="muted">No ML candidates yet. Click <strong>Find speech segments</strong>, or run <code>vad_segments.py</code> / <code>propose_candidates.py</code>.</p>';

  annList.querySelectorAll('.row').forEach(row => {
    const spInput = row.querySelector('input[id^="ann-spk-"]');
    if (spInput) {
      spInput.dataset.field = 'speaker';
      wireSpeakerChips(row.querySelector('.speaker-row'), spInput);
    }
    wireCategoryDetail(row);
    row.querySelectorAll('button[data-act]').forEach(btn => btn.onclick = async () => {
      const uuid = row.dataset.uuid;
      const ann = anns.find(a => a.uuid === uuid);
      const act = btn.dataset.act;
      if (act === 'seek' && ann) {
        pauseIfPlaying();
        const a = (ann.startMs||ann.tMs||0)/1000;
        const b = ann.endMs != null ? ann.endMs/1000 : a + 0.3;
        selStart = a; selEnd = b; normalizeSel();
        setPlayhead(selStart);
        syncSelInputs();
        startBufferPlayback(selStart, selEnd);
        return;
      }
      const tax = readTaxonomyFrom(row);
      if (act === 'confirm') {
        const errMsg = validateTaxonomy(tax);
        if (errMsg) { alert(errMsg); return; }
      }
      try {
        await postJson('/api/annotation/update', {
          kit: k.folder, uuid, action: act,
          ...taxonomyPayload(tax),
        });
        await softRefreshCurrent();
        if (act === 'confirm') flashSaveStatus(`Confirmed → tags.json`);
        if (act === 'dismiss') flashSaveStatus('Dismissed candidate');
      } catch (err) {
        flashSaveStatus('Update failed: ' + err.message, false);
      }
    });
  });
}

async function addTag() {
  const form = document.querySelector('.tag-form');
  const tax = readTaxonomyFrom(form || document);
  let start = parseFloat(document.getElementById('startInput').value);
  let end = parseFloat(document.getElementById('endInput').value);
  const errMsg = validateTaxonomy(tax);
  if (errMsg) { alert(errMsg); return; }
  if (Number.isNaN(start)) start = playheadSec;
  if (Number.isNaN(end)) end = start;
  if (end < start) { const t = start; start = end; end = t; }
  pauseIfPlaying();
  // Park cursor at the snippet start (or current playhead) so Play continues from here.
  const parkAt = Number.isFinite(start) ? start : playheadSec;
  try {
    const data = await postJson('/api/tag/add', {
      kit: current.folder,
      ...taxonomyPayload(tax),
      startMs: Math.round(start * 1000),
      endMs: Math.round(end * 1000),
    });
    document.getElementById('categoryInput').value = '';
    document.getElementById('speakerInput').value = '';
    const wordEl = document.getElementById('wordInput');
    const phoneticEl = document.getElementById('phoneticInput');
    const noteEl = document.getElementById('noteInput');
    if (wordEl) wordEl.value = '';
    if (phoneticEl) phoneticEl.value = '';
    if (noteEl) noteEl.value = '';
    wireSpeakerChips(document.getElementById('addSpeakerRow'), document.getElementById('speakerInput'));
    syncDetailFields(form || document);
    selStart = selEnd = null;
    setPlayhead(parkAt);
    syncSelInputs();
    paintOverlays();
    updatePlayButton();
    await softRefreshCurrent();
    const path = data.tagsPath || current.tagsPath || 'tags.json';
    const n = data.tagCount != null ? data.tagCount : (current.tags || []).length;
    flashSaveStatus(`Saved → ${path} · ${n} tags on disk`);
  } catch (err) {
    flashSaveStatus('Save failed: ' + err.message, false);
  }
}

window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea')) return;
  if (e.code === 'Space') {
    e.preventDefault();
    playToggle();
  }
  if (e.key === '=' || e.key === '+') { e.preventDefault(); zoomBy(0.7); }
  if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomBy(1 / 0.7); }
  if (e.key === '0') { e.preventDefault(); fitAll(); }
  if (e.key === 'Escape') {
    pauseIfPlaying();
    selStart = selEnd = null;
    syncSelInputs();
    paintOverlays();
  }
});
window.addEventListener('resize', () => {
  if (audioBuf) { drawWave(); drawOverview(); paintOverlays(); }
  renderWordCloud();
});

wireWordCloud();
document.getElementById('btnSync').onclick = async () => {
  const btn = document.getElementById('btnSync');
  btn.disabled = true;
  btn.textContent = 'Syncing…';
  flashSaveStatus('USB sync in progress…');
  try {
    const data = await postJson('/api/sync', {});
    if (!data.ok) {
      flashSaveStatus('Sync failed: ' + (data.error || 'iPhone not connected'), false);
      return;
    }
    const pullN = (data.pull && data.pull.pulled || []).length;
    const pushN = (data.push && data.push.pushed || []).length;
    const pushTags = (data.push && data.push.pushed || []).reduce((s, x) => s + (x.tagCount || 0), 0);
    const name = (data.status && data.status.deviceName) || 'iPhone';
    if (pullN === 0 && pushN === 0) {
      flashSaveStatus(
        `Already in sync with ${name}. If a new phone recording is missing: open BabyTalk on the phone (it exports kits to Backups/sync), wait for WAV export to finish, then Sync again.`,
      );
    } else {
      flashSaveStatus(
        `Synced with ${name}: pulled ${pullN} kit(s), pushed ${pushN} kit(s) / ${pushTags} tags. Open BabyTalk on the phone to auto-import tags.`,
      );
    }
    const folder = current && current.folder;
    await refresh();
    if (folder) {
      const i = kits.findIndex(x => x.folder === folder);
      if (i >= 0) await softRefreshCurrent();
    }
  } catch (err) {
    flashSaveStatus('Sync failed: ' + err.message, false);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Sync with iPhone';
  }
};
refresh();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/kits":
            payload = [kit_payload(k) for k in list_kits(ROOT)]
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
            return
        if parsed.path == "/api/sync/status":
            self._send_json(200, run_iphone_sync("status"))
            return
        if parsed.path == "/audio":
            qs = parse_qs(parsed.query)
            name = (qs.get("kit") or [""])[0]
            try:
                kit = resolve_kit(name)
            except FileNotFoundError:
                self._send(404, b"kit not found", "text/plain")
                return
            manifest = load_json(kit / "manifest.json")
            audio = kit / manifest.get("audioFile", "audio.wav")
            if not audio.exists():
                self._send(404, b"missing audio", "text/plain")
                return
            data = audio.read_bytes()
            ctype = mimetypes.guess_type(str(audio))[0] or "audio/wav"
            self._send(200, data, ctype)
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/sync":
            self._send_json(200, run_iphone_sync("sync"))
            return

        try:
            body = self._read_json()
            kit = resolve_kit(body.get("kit", ""))
        except Exception as e:
            self._send(400, str(e).encode("utf-8"), "text/plain")
            return

        if parsed.path == "/api/kit/rename":
            manifest = load_json(kit / "manifest.json")
            new_name = (body.get("sessionName") or "").strip()
            if not new_name:
                self._send(400, b'{"error":"sessionName required"}', "application/json")
                return
            # Preserve the first known human name as originalSessionName.
            prev = (manifest.get("sessionName") or "").strip()
            if not manifest.get("originalSessionName"):
                seed = prev or (manifest.get("filename") or kit.name)
                if seed and seed != new_name:
                    manifest["originalSessionName"] = seed
            manifest["sessionName"] = new_name
            write_json(kit / "manifest.json", manifest)
            self._send_json(200, {"ok": True, "manifest": manifest})
            return

        if parsed.path == "/api/kit/metadata":
            manifest = load_json(kit / "manifest.json")
            # Empty string clears the field; missing key leaves it unchanged.
            for key in (
                "voiceNotesOriginalFilename",
                "recordingLocation",
                "contextNotes",
            ):
                if key in body:
                    val = body.get(key)
                    if val is None:
                        continue
                    text = str(val).strip()
                    if text:
                        manifest[key] = text
                    else:
                        manifest.pop(key, None)
            write_json(kit / "manifest.json", manifest)
            self._send_json(200, {"ok": True, "manifest": manifest})
            return

        if parsed.path == "/api/tag/add":
            tags = read_tags(kit)
            start_ms = int(body.get("startMs") or 0)
            end_ms = body.get("endMs")
            end_ms = int(end_ms) if end_ms is not None else start_ms
            if any(k in body for k in ("category", "speaker", "note", "word", "phonetic")):
                err = taxonomy_validation_error(body)
                if err:
                    self._send_json(400, {"ok": False, "error": err})
                    return
            elif not (body.get("label") or "").strip():
                self._send_json(
                    400, {"ok": False, "error": "category (or legacy label) required"}
                )
                return
            tag = {
                "uuid": str(uuid.uuid4()),
                "label": "untitled",
                "startMs": start_ms,
                "endMs": end_ms,
                "tMs": start_ms,
                "source": "user",
                "status": "confirmed",
            }
            apply_taxonomy_fields(tag, body)
            tags.append(tag)
            write_tags(kit, tags)
            path = str((kit / "tags.json").resolve())
            self._send(
                200,
                json.dumps({"ok": True, "tagsPath": path, "tagCount": len(tags)}).encode(
                    "utf-8"
                ),
                "application/json",
            )
            return

        if parsed.path == "/api/tag/update":
            tags = read_tags(kit)
            for t in tags:
                if t.get("uuid") != body.get("uuid"):
                    continue
                if any(k in body for k in ("category", "speaker", "note", "word", "phonetic")):
                    err = taxonomy_validation_error(body)
                    if err:
                        self._send_json(400, {"ok": False, "error": err})
                        return
                apply_taxonomy_fields(t, body)
                if body.get("startMs") is not None:
                    t["startMs"] = int(body["startMs"])
                    t["tMs"] = t["startMs"]
                if "endMs" in body:
                    t["endMs"] = (
                        int(body["endMs"]) if body["endMs"] is not None else None
                    )
            write_tags(kit, tags)
            path = str((kit / "tags.json").resolve())
            self._send(
                200,
                json.dumps({"ok": True, "tagsPath": path, "tagCount": len(tags)}).encode(
                    "utf-8"
                ),
                "application/json",
            )
            return

        if parsed.path == "/api/tag/delete":
            tags = [t for t in read_tags(kit) if t.get("uuid") != body.get("uuid")]
            write_tags(kit, tags)
            path = str((kit / "tags.json").resolve())
            self._send(
                200,
                json.dumps({"ok": True, "tagsPath": path, "tagCount": len(tags)}).encode(
                    "utf-8"
                ),
                "application/json",
            )
            return

        if parsed.path == "/api/annotation/update":
            anns = read_annotations(kit)
            for a in anns:
                if a.get("uuid") != body.get("uuid"):
                    continue
                action = body.get("action")
                if action == "confirm":
                    if any(
                        k in body
                        for k in ("category", "speaker", "note", "word", "phonetic")
                    ):
                        err = taxonomy_validation_error(body)
                        if err:
                            self._send_json(
                                400,
                                {"ok": False, "error": err},
                            )
                            return
                    apply_taxonomy_fields(a, body)
                    if not (a.get("label") or "").strip():
                        a["label"] = "confirmed"
                    a["status"] = "confirmed"
                    a["source"] = "ml_confirmed"
                    # Also promote into tags.json so phone import of tags sees it.
                    tags = read_tags(kit)
                    existing = next(
                        (t for t in tags if t.get("uuid") == a["uuid"]), None
                    )
                    payload = {
                        "uuid": a["uuid"],
                        "label": a["label"],
                        "startMs": a.get("startMs", a.get("tMs", 0)),
                        "endMs": a.get("endMs"),
                        "tMs": a.get("startMs", a.get("tMs", 0)),
                        "source": "ml_confirmed",
                        "status": "confirmed",
                    }
                    for key in ("category", "speaker", "word", "phonetic", "note"):
                        if a.get(key):
                            payload[key] = a[key]
                        elif existing and key in existing:
                            existing.pop(key, None)
                    if existing:
                        # Drop stale detail fields when category flips.
                        for key in ("word", "phonetic", "note"):
                            if key not in payload:
                                existing.pop(key, None)
                        existing.update(payload)
                    else:
                        tags.append(payload)
                    write_tags(kit, tags)
                elif action == "dismiss":
                    a["status"] = "dismissed"
            write_annotations(kit, anns)
            self._send(200, b'{"ok":true}', "application/json")
            return

        if parsed.path == "/api/vad/run":
            result = run_vad_for_kit(kit, body)
            code = 200 if result.get("ok") else 500
            # Avoid dumping every annotation uuid into the HTTP response on large kits.
            slim = {
                k: v
                for k, v in result.items()
                if k != "annotations"
            }
            if result.get("ok"):
                slim["sample"] = (result.get("annotations") or [])[:5]
            self._send_json(code, slim)
            return

        # Back-compat alias
        if parsed.path == "/api/update":
            body["action"] = body.get("action") or "confirm"
            self.path = "/api/annotation/update"
            return self.do_POST()

        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def run_iphone_sync(command: str) -> dict:
    """Run tools/iphone_sync.py via the tools venv when available."""
    script = TOOLS_DIR / "iphone_sync.py"
    py = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
    if not script.exists():
        return {"ok": False, "error": f"Missing {script}"}
    try:
        proc = subprocess.run(
            [str(py), str(script), command, "--json", "--bundle-id", BUNDLE_ID],
            capture_output=True,
            text=True,
            timeout=60 * 30,
            cwd=str(TOOLS_DIR.parent),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Sync timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    raw = (proc.stdout or "").strip()
    if not raw:
        err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
        return {"ok": False, "error": err}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": (proc.stderr or raw)[:500],
        }
    if command == "status":
        data.setdefault("ok", bool(data.get("connected")))
    return data


def main(argv: list[str]) -> int:
    global ROOT, BUNDLE_ID
    ensure_library()
    seeded = seed_library_from_backups()
    if len(argv) >= 2 and not argv[1].startswith("-"):
        ROOT = Path(argv[1]).expanduser().resolve()
    else:
        ROOT = LIBRARY_DIR.resolve()
    if not ROOT.exists():
        print(f"Path not found: {ROOT}")
        return 1
    port = 8765
    # Optional: review_server.py [path] [port] or review_server.py [port]
    if len(argv) >= 3:
        port = int(argv[2])
    elif len(argv) == 2 and argv[1].isdigit():
        port = int(argv[1])
        ROOT = LIBRARY_DIR.resolve()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Review UI: http://127.0.0.1:{port}")
    print(f"Kits root: {ROOT}")
    if seeded:
        print(f"Seeded {seeded} kit(s) into {LIBRARY_DIR}")
    print(f"Library: {LIBRARY_DIR}")
    print("Sync with iPhone uses USB (pymobiledevice3). Open the app on the phone to auto-import tags.")
    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401
        print("Speech VAD: ready (numpy + soundfile). Use Find speech segments in the UI.")
    except ImportError:
        print(
            "Speech VAD: install deps first — "
            "tools/.venv/bin/pip install numpy soundfile"
        )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
