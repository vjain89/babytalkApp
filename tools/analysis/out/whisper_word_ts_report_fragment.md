## Whisper word timestamps vs syllable over-splits

### Problem (plain language)

The current ML path (VAD → ECAPA → pause-split → DJW resegment) often
**cuts a single spoken word into syllable-sized boxes** (e.g. `tea|cher`).
Those too-short children fail IoU≥0.5 against human word tags even when
the sound was detected. IoU sweeps showed knob-tuning does not fix this —
the lever is **merge-back** (and/or an external word-boundary signal).

### What Whisper word timestamps are

`faster-whisper` can emit not only a transcript string but **per-word
start/end times** (`word_timestamps=True`). Model used here:
**faster-whisper/base** (override with `BABYTALK_WHISPER_MODEL`;
same default as `tools/asr_suggest.py`).

Method: cluster nearby manual tags into padded audio windows, transcribe
each window once with word timestamps, align Whisper word intervals to
human tags by overlap (and separately by fuzzy text). Boxes are **not**
forced to human edges — this answers “if we used Whisper word boxes
instead of DJW children, how well would they match tags?”

Eval script: `tools/analysis/whisper_word_ts_eval.py`  
Full JSON: `tools/analysis/out/whisper_word_ts.json`

### Why they could help

If Whisper’s word intervals landed near human word spans, we could use
them as candidate boxes (or as merge targets for DJW children) instead of
trusting syllable cuts alone — even when **spelling** is wrong for Swiss
German / baby speech. Boundary quality (overlap IoU) is the decision
metric; text match is reported separately and expected to be weak.

### Measured metrics

Kits: `26_07_27__19:53:00` (St_87, English-leaning), `26_07_05__00:00:00`
(St_75, Swiss German). Manual verbal tags with a `word` label; excluded
`non-verbal vocalization` and `source=ml_confirmed`. **n=179** pooled.

| Kit | n | Any overlap | Text match | IoU≥0.5 (overlap) | Med IoU (hits) | Archival ML IoU≥0.5 |
|---|---:|---:|---:|---:|---:|---:|
| 26_07_27__19:53:00 | 107 | 95.3% | 11.2% | **30.8%** | 0.433 | 48.6% |
| 26_07_05__00:00:00 | 72 | 93.1% | 5.6% | **18.1%** | 0.387 | 11.1%† |
| **POOLED** | **179** | **94.4%** | **8.9%** | **25.7%** | **0.416** | — |

† Archival `annotations.json` on St_75 looks stale/too-short vs prior
**iou_sweep** fresh DJW (~53.8% manual IoU≥0.5 on that kit; pooled sweep
~57%). Prefer sweep as DJW reference, not raw archival on St_75.

| Pooled error mode | % of tags | Meaning |
|---|---:|---|
| word_split | **40.8%** | ≥2 Whisper tokens overlap one human word; best single-token IoU&lt;0.5 |
| wrong_word | **36.9%** | Overlap exists but fuzzy text ≠ tag (often still mediocre IoU) |
| big_boundary_error | 7.3% | Overlap but IoU&lt;0.3 |
| no_speech | 5.6% | No Whisper word overlaps the tag |
| boundary_soft | 3.9% | Text ok-ish, IoU in (0.3, 0.5) |
| ok | 2.8% | Text + IoU≥0.5 |
| word_merge | 2.8% | One long Whisper token covering a short tag poorly |

**Qualitative failure modes**

- **ASR spelling is useless as a gate:** only 8.9% fuzzy text match
  (`teacher`→`Take`/`a...`, `schuhe`→`we're...`, `fuchs`→`Das`). Swiss
  German kit is worse (5.6%).
- **Whisper over-splits too:** dominant mode is `word_split` (40.8%). It
  does **not** reliably emit one box per human word — same failure class
  as DJW syllables, just with different edges.
- **Occasional good boundaries with wrong text:** e.g. `teacher`↔`a...`
  IoU ≈0.72 — so Whisper *can* hit duration, but not often enough
  (pooled IoU≥0.5 only 25.7% vs DJW ~50–57%).
- **Rare clean wins:** short closed-class / clear tokens (`do`, `Jacke`).

### Recommendation

**Need DJW merge-back — Whisper not enough alone**

Overlap coverage is high (94%), but **word-box IoU≥0.5 is only 25.7%** —
worse than DJW/fresh candidates on the same problem (~50–57% from
iou_sweep). Text match is ~9%, so Whisper cannot be the merge oracle by
lexeme either. Prioritize **acoustic DJW merge-back** for syllable
over-splits; treat Whisper word timestamps as an optional weak prior only
where a single high-confidence token already IoU-aligns — not as a
replacement for word-level boxes.

_Hybrid later?_ Only if merge-back plateaus and a larger Whisper model or
forced-alignment (lexicon-constrained) is tested; `base` word timestamps
as evaluated here are not the IoU lever.

_Generated 2026-07-29T22:27:08+00:00_
