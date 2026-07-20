"""Minimal Mac-side review queue for provisional annotations (P3).

Serves a local web UI over a session-kit folder (or backup batch). Confirm /
dismiss / relabel candidates; writes annotations.json back into each kit.

Usage:
  python3 tools/review_server.py /path/to/Backups/2026-07-20_1200
  # open http://127.0.0.1:8765

Requires: only Python 3 stdlib.
"""

from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT: Path = Path(".")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_kits(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()])


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
    return {
        "folder": kit.name,
        "manifest": manifest,
        "tags": tags,
        "annotations": anns,
        "audioUrl": f"/audio?kit={kit.name}",
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>BabyTalk Review</title>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #f6f4f1; color: #1c1c1c; }
  header { padding: 16px 20px; background: #1c1c1c; color: #f6f4f1; }
  main { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 64px); }
  #kits { border-right: 1px solid #ddd; padding: 12px; overflow: auto; }
  #kits button { display: block; width: 100%; text-align: left; margin: 4px 0; padding: 8px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
  #kits button.active { border-color: #1c1c1c; background: #eee; }
  #panel { padding: 16px 20px; }
  audio { width: 100%; margin: 12px 0; }
  .row { display: flex; gap: 8px; align-items: center; padding: 8px 0; border-bottom: 1px solid #e5e5e5; }
  .row input[type=text] { flex: 1; padding: 6px; }
  .muted { color: #666; font-size: 13px; }
  .pill { font-size: 11px; padding: 2px 6px; background: #e8e0d5; border-radius: 4px; }
</style>
</head>
<body>
<header><strong>BabyTalk</strong> · Mac review queue (provisional → confirmed)</header>
<main>
  <aside id="kits"></aside>
  <section id="panel"><p class="muted">Select a session kit.</p></section>
</main>
<script>
let kits = [];
let current = null;

async function refresh() {
  const res = await fetch('/api/kits');
  kits = await res.json();
  const el = document.getElementById('kits');
  el.innerHTML = kits.map((k,i) =>
    `<button data-i="${i}">${k.folder}<br/><span class="muted">${(k.manifest.durationMs/1000).toFixed(1)}s · ${k.annotations.filter(a=>a.status!=='confirmed').length} open</span></button>`
  ).join('');
  el.querySelectorAll('button').forEach(b => b.onclick = () => select(+b.dataset.i));
}

function select(i) {
  current = kits[i];
  document.querySelectorAll('#kits button').forEach((b,idx) => b.classList.toggle('active', idx===i));
  render();
}

function render() {
  const k = current;
  if (!k) return;
  const open = k.annotations.filter(a => a.status !== 'dismissed');
  document.getElementById('panel').innerHTML = `
    <h2>${k.manifest.sessionName || k.folder}</h2>
    <p class="muted">${k.manifest.recordingUuid} · hash ${String(k.manifest.audioContentHash||'').slice(0,12)}…</p>
    <audio controls src="${k.audioUrl}"></audio>
    <h3>Candidates</h3>
    ${open.map((a,idx) => `
      <div class="row" data-uuid="${a.uuid}">
        <span class="pill">${(a.startMs/1000).toFixed(2)}s${a.endMs!=null?('–'+(a.endMs/1000).toFixed(2)+'s'):''}</span>
        <input type="text" value="${(a.label||'').replace(/"/g,'&quot;')}" placeholder="free-form label"/>
        <button data-act="confirm">Confirm</button>
        <button data-act="dismiss">Dismiss</button>
        <button data-act="seek">Seek</button>
      </div>`).join('') || '<p class="muted">No candidates. Run propose_candidates.py first.</p>'}
    <h3>User tags</h3>
    <p class="muted">${k.tags.map(t => t.label + '@' + (t.startMs/1000).toFixed(1) + 's').join(' · ') || 'none'}</p>
  `;
  document.querySelectorAll('.row').forEach(row => {
    row.querySelectorAll('button').forEach(btn => btn.onclick = async () => {
      const uuid = row.dataset.uuid;
      const act = btn.dataset.act;
      const label = row.querySelector('input').value.trim();
      if (act === 'seek') {
        const ann = k.annotations.find(a => a.uuid === uuid);
        const audio = document.querySelector('audio');
        if (ann && audio) { audio.currentTime = (ann.startMs||0)/1000; audio.play(); }
        return;
      }
      await fetch('/api/update', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ kit: k.folder, uuid, action: act, label })
      });
      await refresh();
      const idx = kits.findIndex(x => x.folder === k.folder);
      if (idx >= 0) select(idx);
    });
  });
}

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
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/kits":
            payload = [kit_payload(k) for k in list_kits(ROOT)]
            data = json.dumps(payload).encode("utf-8")
            self._send(200, data, "application/json")
            return
        if parsed.path == "/audio":
            qs = parse_qs(parsed.query)
            name = (qs.get("kit") or [""])[0]
            kit = ROOT / name if name != ROOT.name else ROOT
            if name and (ROOT / name).exists():
                kit = ROOT / name
            elif (ROOT / "manifest.json").exists():
                kit = ROOT
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
        if parsed.path != "/api/update":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        kit = ROOT / body["kit"]
        if not kit.exists() and (ROOT / "manifest.json").exists():
            kit = ROOT
        ann_path = kit / "annotations.json"
        payload = load_json(ann_path) if ann_path.exists() else {"annotations": []}
        anns = payload.get("annotations", [])
        for a in anns:
            if a.get("uuid") != body["uuid"]:
                continue
            if body["action"] == "confirm":
                a["label"] = body.get("label") or a.get("label") or "confirmed"
                a["status"] = "confirmed"
                a["source"] = "ml_confirmed"
            elif body["action"] == "dismiss":
                a["status"] = "dismissed"
        write_json(ann_path, {"annotations": anns})
        self._send(200, b'{"ok":true}', "application/json")

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main(argv: list[str]) -> int:
    global ROOT
    if len(argv) < 2:
        print(__doc__)
        return 2
    ROOT = Path(argv[1]).expanduser().resolve()
    if not ROOT.exists():
        print(f"Path not found: {ROOT}")
        return 1
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print(f"Review UI: http://127.0.0.1:8765")
    print(f"Kits root: {ROOT}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
