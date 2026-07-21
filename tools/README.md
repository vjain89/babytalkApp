# BabyTalk Mac tools

## Review studio (tag spans in the browser)

1. On the iPhone: **Prepare USB Backup** (writes `Documents/Backups/<date>/` session kits).
2. Plug into Mac → Finder → your iPhone → babytalkApp → copy the backup folder to the Mac.
3. Run:

```bash
python3 tools/review_server.py /path/to/Backups/<date>
```

4. Open http://127.0.0.1:8765
5. Select a session → play → **drag on the waveform** to select a span → enter a label → **Add tag**.
6. Tags are written into each kit’s `tags.json` (with `startMs` / `endMs`).
7. Copy the reviewed kit folder(s) back into the phone’s `Documents/Import/`.
8. On the iPhone: **Import Inbox Annotations**.

Optional ML candidates:

```bash
pip install numpy soundfile
python3 tools/propose_candidates.py /path/to/Backups/<date>
python3 tools/validate_export.py /path/to/Backups/<date>
```

Then reopen the review UI to confirm/dismiss proposals.
